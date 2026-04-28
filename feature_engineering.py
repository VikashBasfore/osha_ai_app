import pandas as pd
import numpy as np
def create_features(df):

    df = df.copy()

    ### Columns we Should Map 

    # 1. incident_outcome
    if 'incident_outcome' in df.columns:
        df['incident_outcome'] = df['incident_outcome'].map({
            1: 'Death',
            2: 'Days Away From Work',
            3: 'Job Transfer/Restriction',
            4: 'Other Case'
        })

    # 2. type_of_incident
    df['type_of_incident'] = df['type_of_incident'].map({
        1: 'Injury',
        2: 'Skin Disorder',
        3: 'Respiratory Condition',
        4: 'Poisoning',
        5: 'Hearing Loss',
        6: 'Other Illness'
    })

    # 3. soc_reviewed
    df['soc_reviewed'] = df['soc_reviewed'].map({
        0: 'Not Reviewed',
        1: 'Reviewed by OSHA',
        2: 'Not SOC coded'
    })

    # 4. time_unknown
    df['time_unknown'] = df['time_unknown'].map({
        0: 'Known Time',
        1: 'Unknown Time'
    })

    # 5 establishment_type
    df['establishment_type'] = df['establishment_type'].map({
        1: 'Private Industry',
        2: 'State Government',
        3: 'Local Government'
    })
    
    # Convert incident year and month to numeric format
    df['incident_year'] = df['incident_year'].astype(int)
    df['incident_month'] = df['incident_month'].astype(int)

    # Extract day of week
    df['incident_day'] = df['date_of_incident'].dt.dayofweek

    # Shift
    df['shift'] = pd.cut(
        df['incident_hour'],
        bins=[0,6,12,18,24],
        labels=['Night','Morning','Afternoon','Evening'],
        include_lowest=True
    )

    # Size category
    df['size_category'] = df['size'].map({
        1: 'Small',
        21: 'Medium',
        22: 'Medium',
        2: 'Medium',
        3: 'Large'
    })

    # Hours after start
    df['hours_after_start'] = df['incident_hour'] - df['start_hour']

    # Industry sector
    df['industry_sector'] = df['naics_code'].astype(str).str[:2]

    # Weekend
    df['is_weekend'] = df['date_of_incident'].dt.weekday >= 5
    df['is_weekend'] = df['is_weekend'].astype(int)

    # Working hour
    df['working_hour'] = df['incident_hour'].apply(
        lambda x: 1 if 6 <= x <= 18 else 0
    )

    # Season
    df['season'] = pd.cut(
        df['incident_month'],
        bins=[0,3,6,9,12],
        labels=['Winter','Spring','Summer','Fall']
    )

    # SOC group
    df['soc_group'] = df['soc_code'].str[:2]

    # Rare sector grouping
    sector_counts = df['industry_sector'].value_counts()
    rare_sectors = sector_counts[sector_counts < 100].index
    df['industry_sector'] = df['industry_sector'].replace(rare_sectors, 'Other')

    # Severity
    df['severity_score'] = df['dafw_num_away'] + df['djtr_num_tr']
    
    df['employees_log'] = np.log1p(df['annual_average_employees'])
    df['hours_log'] = np.log1p(df['total_hours_worked'])
    df['soc_prob_log'] = np.log1p(df['soc_probability'])

# DROP ORIGINAL
    df.drop(columns=[
        'annual_average_employees',
        'total_hours_worked',
        'soc_probability'
    ], inplace=True)

# DROP UNUSED
    df = df.drop(columns=['dafw_num_away', 'djtr_num_tr'])

    return df

class FeatureEngineer:

    def __init__(self):
        self.rare_sectors = None

    def fit(self, df):
        sector_counts = df['industry_sector'].value_counts()
        self.rare_sectors = sector_counts[sector_counts < 100].index
        return self

    def transform(self, df):

        df = df.copy()

        # APPLY RARE GROUPING 🔥
        df['industry_sector'] = df['industry_sector'].replace(self.rare_sectors, 'Other')

        return df