import pandas as pd
from src.config import RAW_DATA_PATH

def load_raw_data():
    """
    Loads the marketing campaign dataset using the path 
    defined in the config file.
    """
    # Load the data using the tab separator
    df = pd.read_csv(RAW_DATA_PATH, sep='\t')
    return df