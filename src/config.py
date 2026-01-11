import os

# 1. Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "marketing_campaign.csv")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "cleaned_data.csv")

# 2. ML Constants
RANDOM_STATE = 42
CV_FOLDS = 5

# 3. Feature Groups (The missing pieces!)
MNT_COLS = [
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

CAMPAIGN_COLS = [
    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
    'AcceptedCmp4', 'AcceptedCmp5', 'Response'
]