import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Encoder:

    def __init__(self):
        self.freq_maps = {}
        self.label_encoder = LabelEncoder()
        self.columns = None

    def fit(self, df):

        df = df.copy()

        # DROP
        df = df.drop(columns=[
            'date_of_incident',
            'establishment_name',
            'street_address',
            'city',
            'naics_code'
        ])

        # STORE FREQUENCY MAPS 🔥
        for col in ['industry_description', 'soc_description', 'company_name', 'soc_code']:
            self.freq_maps[col] = df[col].value_counts()

        # LABEL ENCODER
        self.label_encoder.fit(df['incident_outcome'])

        # APPLY ON TRAIN TO LOCK COLUMNS
        df = self.transform(df)
        self.columns = df.columns

        return self

    def transform(self, df):

        df = df.copy()

        df = df.drop(columns=[
            'date_of_incident',
            'establishment_name',
            'street_address',
            'city',
            'naics_code'
        ], errors='ignore')

        # APPLY STORED FREQUENCY 🔥
        for col, freq in self.freq_maps.items():
            df[col] = df[col].map(freq).fillna(0)

        # ONE HOT
        df = pd.get_dummies(
            df,
            columns=[
                'shift','size_category','season','establishment_type',
                'soc_reviewed','time_unknown','industry_sector',
                'type_of_incident','naics_year','soc_group'
            ],
            drop_first=True,
            sparse=True
        )

        # APPLY SAME LABEL ENCODER 🔥
        if 'incident_outcome' in df.columns:
            df['incident_outcome'] = self.label_encoder.transform(df['incident_outcome'])

        # ALIGN COLUMNS
        if self.columns is not None:
            for col in self.columns:
                if col not in df:
                    df[col] = 0

            df = df[self.columns]

        return df