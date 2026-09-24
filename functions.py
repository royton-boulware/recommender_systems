import pandas as pd
import os
import snowflake.connector
from snowflake.connector.errors import ProgrammingError
from snowflake.connector.errors import ProgrammingError
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, dsa
from cryptography.hazmat.primitives import serialization


def _get_private_key() -> bytes:
    """Helper function to load and decode the private key from the environment."""
    key_path = os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH')
    passphrase = os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE')
    
    if not key_path or not os.path.exists(key_path):
        raise FileNotFoundError(f"Private key file not found at: {key_path}")

    with open(key_path, "rb") as key_file:
        p_key = serialization.load_pem_private_key(
            key_file.read(),
            password=passphrase.encode() if passphrase else None,
            backend=default_backend()
        )

    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

def query_snowflake_to_df(query: str, params: tuple = None) -> pd.DataFrame:
    """
    Connects to Snowflake, executes a query, and returns the results 
    as a pandas DataFrame.
    
    Args:
        query (str): The SQL query to execute.
        params (tuple, optional): Parameters to bind to the query.
        
    Returns:
        pd.DataFrame: A DataFrame containing the query results. 
                      Returns an empty DataFrame on error.
    """
    pkb = _get_private_key()
    
    try:
        with snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            private_key=pkb,
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            role=os.getenv('SNOWFLAKE_ROLE')
        ) as conn:
            
            with conn.cursor() as cur:
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                
                # Fetch results directly into a pandas DataFrame using Arrow
                df = cur.fetch_pandas_all()
                
                # Optional: Snowflake returns column names in uppercase by default. 
                # Uncomment the line below if you prefer lowercase column names.
                # df.columns = df.columns.str.lower()
                
                return df
                
    except ProgrammingError as e:
        print(f"Snowflake Programming Error: {e}")
        return pd.DataFrame() # Return empty DataFrame so downstream code doesn't break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()