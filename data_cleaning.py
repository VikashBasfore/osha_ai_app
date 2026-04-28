import pandas as pd

def clean_data(df):

    df = df.copy()

    # DATE
    if 'date_of_incident' in df.columns:
        df['date_of_incident'] = pd.to_datetime(df['date_of_incident'], errors='coerce')
        df['incident_year'] = df['date_of_incident'].dt.year
        df['incident_month'] = df['date_of_incident'].dt.month

    # DROP USELESS
    cols_to_drop = ['id','establishment_id','ein','case_number','created_timestamp']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # DROP IMPORTANT NULLS
    important_cols = [
        'soc_code','soc_description','soc_probability',
        'soc_reviewed','date_of_incident'
    ]
    existing_cols = [c for c in important_cols if c in df.columns]
    if existing_cols:
        df = df.dropna(subset=existing_cols)

    # DROP HIGH NULL
    df.drop(columns=['date_of_death'], errors='ignore', inplace=True)

    # TEXT FILL
    for col in [
        'NEW_INCIDENT_DESCRIPTION',
        'NEW_NAR_BEFORE_INCIDENT',
        'NEW_NAR_WHAT_HAPPENED'
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # MODE FILL
    for col in ['NEW_INCIDENT_LOCATION','NEW_NAR_INJURY_ILLNESS']:
        if col in df.columns and not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    # COMPANY
    if 'company_name' in df.columns:
        df['company_name'] = df['company_name'].fillna("Unknown Company")

    # NAICS → INDUSTRY
    if 'industry_description' in df.columns and 'naics_code' in df.columns:
        industry_map = df.dropna(subset=['industry_description']) \
            .drop_duplicates('naics_code') \
            .set_index('naics_code')['industry_description'] \
            .to_dict()

        df['industry_description'] = df['industry_description'].fillna(
            df['naics_code'].map(industry_map)
        )

        df = df.dropna(subset=['industry_description'])

    # TIME UNKNOWN
    if 'time_unknown' in df.columns and not df['time_unknown'].mode().empty:
        df['time_unknown'] = df['time_unknown'].fillna(df['time_unknown'].mode()[0])

    # JOB CLEAN
    if 'job_description' in df.columns:
        df['job_description'] = df['job_description'].fillna("Unknown").str.lower().str.strip()

    # INCIDENT TIME
    if 'time_of_incident' in df.columns:
        df['time_of_incident'] = df['time_of_incident'].astype(str).str.replace('.000','', regex=False)
        df['time_of_incident'] = pd.to_datetime(df['time_of_incident'], errors='coerce')

        df['incident_hour'] = df['time_of_incident'].dt.hour
        df['incident_hour'] = df['incident_hour'].fillna(df['incident_hour'].median())

        df.drop(columns=['time_of_incident'], inplace=True, errors='ignore')

    # OBJECT CLEAN
    if 'NEW_NAR_OBJECT_SUBSTANCE' in df.columns:
        df['NEW_NAR_OBJECT_SUBSTANCE'] = df['NEW_NAR_OBJECT_SUBSTANCE'].replace(
            ['N A','blank','NA','None','null'], 'Unknown'
        ).fillna('Unknown').str.lower().str.strip()

    # ESTABLISHMENT TYPE (FIXED 🔥)
    if 'establishment_type' in df.columns:
        df['establishment_type'] = df['establishment_type'].fillna(
            df['establishment_type'].mode()[0]
        )

    # START TIME
    if 'time_started_work' in df.columns:
        df['time_started_work'] = df['time_started_work'].astype(str).str.replace('.000','', regex=False)
        df['time_started_work'] = pd.to_datetime(df['time_started_work'], errors='coerce')

        df['start_hour'] = df['time_started_work'].dt.hour
        df['start_hour'] = df['start_hour'].fillna(df['start_hour'].median())

        df.drop(columns=['time_started_work'], inplace=True, errors='ignore')

    # DUPLICATES
    df.drop_duplicates(inplace=True)

    # DROP CONSTANT
    df.drop(columns=['year_filing_for'], errors='ignore', inplace=True)

    # TYPE CONVERSION
    cat_cols = [
        'zip_code','naics_code','naics_year','soc_reviewed',
        'incident_outcome','type_of_incident','time_unknown',
        'incident_year','incident_month','establishment_type'
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')

    # OUTLIERS
    if 'annual_average_employees' in df.columns:
        df['annual_average_employees'] = df['annual_average_employees'].clip(
            upper=df['annual_average_employees'].quantile(0.99)
        )

    if 'total_hours_worked' in df.columns:
        df['total_hours_worked'] = df['total_hours_worked'].clip(
            upper=df['total_hours_worked'].quantile(0.99)
        )

    return df