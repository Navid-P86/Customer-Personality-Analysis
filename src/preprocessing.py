import pandas as pd
import numpy as np

def clean_and_feature_engineer(df):
    """
    Standard cleaning pipeline for the marketing campaign dataset.
    """
    df = df.copy()

    # 1. Handle Missing Values
    df = df.dropna(subset=['Income'])

    # 2. Date Conversion
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

    # 3. Feature Engineering: Demographics
    df['Age'] = 2015 - df['Year_Birth']
    df['Children'] = df['Kidhome'] + df['Teenhome']
    df['Is_Parent'] = (df['Children'] > 0).astype(int)

    # 4. Feature Engineering: Total Spending
    mnt_cols = [c for c in df.columns if 'Mnt' in c]
    df['TotalSpending'] = df[mnt_cols].sum(axis=1)

    # 5. Simplify Categories
    df['Education'] = df['Education'].replace({
        'Basic': 'Undergrad', '2n Cycle': 'Undergrad',
        'Graduation': 'Graduate', 'Master': 'Postgrad', 'PhD': 'Postgrad'
    })
    
    df['Marital_Status'] = df['Marital_Status'].replace({
        'Married':'Partner', 'Together':'Partner', 'Absurd':'Alone', 
        'Widow':'Alone', 'YOLO':'Alone', 'Divorced':'Alone', 'Single':'Alone'
    })

    # 6. Outlier Removal (standard for this dataset)
    df = df[(df['Age'] < 90) & (df['Income'] < 200000)]

    # 7. Drop redundant columns
    cols_to_drop = ['Z_CostContact', 'Z_Revenue', 'ID', 'Year_Birth']
    df = df.drop(columns=cols_to_drop)

    return df