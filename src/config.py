import os

# 1. Get the path of the folder this file is in (src/)
# 2. Go up one level to get the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 3. Define standard paths
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "marketing_campaign.csv")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "cleaned_data.csv")

# 4. Define Column Groups (Used later in analysis)
MNT_COLS = [
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

CAMPAIGN_COLS = [
    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
    'AcceptedCmp4', 'AcceptedCmp5', 'Response'
]