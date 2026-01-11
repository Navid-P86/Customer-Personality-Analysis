import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "marketing_campaign.csv")

# ML Constants
RANDOM_STATE = 42
CV_FOLDS = 5

# Feature Groups
MNT_COLS = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']