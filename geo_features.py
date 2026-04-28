
import pandas as pd

def add_geo(df):

    df = df.copy()

    # 🚨 IF zip_code not present → skip geo features
    if 'zip_code' not in df.columns:
        return df

    # Load zip data
    zip_coords = pd.read_csv("C:\\Users\\vikas\\Downloads\\uszips.xls")

    zip_coords = zip_coords[['zip','lat','lng']]
    zip_coords.columns = ['zip_code','latitude','longitude']

    # Convert to string
    df['zip_code'] = df['zip_code'].astype(str)
    zip_coords['zip_code'] = zip_coords['zip_code'].astype(str)

    # Merge
    df = df.merge(zip_coords, on='zip_code', how='left')

    # Fill missing
    df['latitude'] = df['latitude'].fillna(df['latitude'].median())
    df['longitude'] = df['longitude'].fillna(df['longitude'].median())

    # Drop zip_code
    df = df.drop(columns=['zip_code'], errors='ignore')

    # Drop state if exists
    df = df.drop(columns=['state'], errors='ignore')

    return df