import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclasses.dataclass
class LlamaConfig:
    """Define Llama model hyperparameters."""
    vocab_size: int = 50000  # Size of the tokenizer vocabulary
    max_position_embeddings: int = 2048  # Maximum sequence length
    hidden_size: int = 768  # Dimension of hidden layers
    intermediate_size: int = 4*768  # Dimension of MLP's hidden layer
    num_hidden_layers: int = 12  # Number of transformer layers
    num_attention_heads: int = 12  # Number of attention heads
    num_key_value_heads: int = 3  # Number of key-value heads for GQA


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input.

    This is a helper function for rotary position embeddings (RoPE).
    For a tensor of shape (..., d), it returns a tensor where the last
    d/2 dimensions are rotated by swapping and negating.

    Args:
        x: Input tensor of shape (..., d)

    Returns:
        Tensor of same shape with rotated last dimension
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)  # Concatenate with rotation


class RotaryPositionEncoding(nn.Module):
    """Rotary position encoding."""

    def __init__(self, dim: int, max_position_embeddings: int) -> None:
        """Initialize the RotaryPositionEncoding module

        Args:
            dim: The hidden dimension of the input tensor to which RoPE is applied
            max_position_embeddings: The maximum sequence length of the input tensor
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        # compute a matrix of n\theta_i
        N = 10_000.0
        inv_freq = 1.0 / (N ** (torch.arange(0, dim, 2) / dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(max_position_embeddings)
        sinusoid_inp = torch.outer(position, inv_freq)
        # save cosine and sine matrices as buffers, not parameters
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x: Tensor) -> Tensor:
        """Apply RoPE to tensor x

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_heads, head_dim)

        Returns:
            Output tensor of shape (batch_size, seq_length, num_heads, head_dim)
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        dtype = x.dtype
        # transform the cosine and sine matrices to 4D tensor and the same dtype as x
        cos = self.cos.to(dtype)[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin.to(dtype)[:seq_len].view(1, seq_len, 1, -1)
        # apply RoPE to x
        output = (x * cos) + (rotate_half(x) * sin)
        return output


class LlamaAttention(nn.Module):
    """Grouped-query attention with rotary embeddings."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads  # GQA: H_kv < H_q

        # hidden_size must be divisible by num_heads
        assert (self.head_dim * self.num_heads) == self.hidden_size

        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, rope: RotaryPositionEncoding, attn_mask: Tensor) -> Tensor:
        bs, seq_len, dim = hidden_states.size()

        # Project inputs to Q, K, V
        query_states = self.q_proj(hidden_states).view(bs, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bs, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bs, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary position embeddings
        query_states = rope(query_states)
        key_states = rope(key_states)

        # Transpose tensors from BSHD to BHSD dimension for scaled_dot_product_attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Use PyTorch's optimized attention implementation
        # setting is_causal=True is incompatible with setting explicit attention mask
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=0.0,
            enable_gqa=True,
        )

        # Transpose output tensor from BHSD to BSHD dimension, reshape to 3D, and then project output
        attn_output = attn_output.transpose(1, 2).reshape(bs, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        # Two parallel projections for SwiGLU
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = F.silu  # SwiGLU activation function
        # Project back to hidden size
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU activation: multiply gate and up-projected inputs
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    """Single transformer layer for a Llama model."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states: Tensor, rope: RotaryPositionEncoding, attn_mask: Tensor) -> Tensor:
        # First residual block: Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attn(hidden_states, rope=rope, attn_mask=attn_mask)
        hidden_states = attn_outputs + residual

        # Second residual block: MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states


class LlamaModel(nn.Module):
    """The full Llama model without any pretraining heads."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.rotary_emb = RotaryPositionEncoding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)

    def forward(self, input_ids: Tensor, attn_mask: Tensor) -> Tensor:
        # Convert input token IDs to embeddings
        hidden_states = self.embed_tokens(input_ids)
        # Process through all transformer layers, then the final norm layer
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope=self.rotary_emb, attn_mask=attn_mask)
        hidden_states = self.norm(hidden_states)
        # Return the final hidden states
        return hidden_states


class LlamaForPretraining(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.base_model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, attn_mask: Tensor) -> Tensor:
        hidden_states = self.base_model(input_ids, attn_mask)
        return self.lm_head(hidden_states)


def create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Create a causal mask for self-attention.

    Args:
        seq_len: Length of the sequence
        device: Device to create the mask on
        dtype: Data type of the mask

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype) \
                .triu(diagonal=1)
    return mask

def create_padding_mask(batch, padding_token_id, device: torch.device, dtype: torch.dtype = torch.float32):
    """Create a padding mask for a batch of sequences for self-attention.

    Args:
        batch: Batch of sequences, shape (batch_size, seq_len)
        padding_token_id: ID of the padding token

    Returns:
        Padding mask of shape (batch_size, 1, seq_len, seq_len)
    """
    padded = torch.zeros_like(batch, device=device, dtype=dtype) \
                  .masked_fill(batch == padding_token_id, float('-inf'))
    mask = padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]


# Create model with default config
model_config = LlamaConfig()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = LlamaForPretraining(model_config).to(device)
# print the model size
print(f"Model parameters size: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
print(f"Model buffers size: {sum(p.numel() for p in model.buffers())/1e6:.2f} M")

# Create a random tensor
PAD_TOKEN_ID = 0
bs, seq_len = 5, 13
x = torch.randint(1, model_config.vocab_size, (bs, seq_len), dtype=torch.int32, device=device)
# set random length of padding tokens at the end of each sequence
for i, pad_length in enumerate([4, 1, 0, 3, 8]):
    if pad_length > 0:
        x[i, -pad_length:] = PAD_TOKEN_ID
# Create causal and padding masks
causal_mask = create_causal_mask(seq_len, device)
padding_mask = create_padding_mask(x, PAD_TOKEN_ID, device)
attn_mask = causal_mask + padding_mask
print(f"Input ids: {x}")
print(f"Attention mask: {attn_mask}")

# Run the model
output = model(x, attn_mask)
print("OK")