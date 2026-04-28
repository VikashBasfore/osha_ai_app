from data_cleaning import clean_data
from feature_engineering import create_features, FeatureEngineer
from encoding import Encoder
from text_processing import TextProcessor
from geo_features import add_geo


class FullPipeline:

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.encoder = Encoder()
        self.text = TextProcessor()

    def fit(self, df):

        df = clean_data(df)
        df = create_features(df)
        df = add_geo(df)

        # 🔥 FIT RARE SECTORS
        self.feature_engineer.fit(df)
        df = self.feature_engineer.transform(df)

        # 🔥 FIT ENCODER
        self.encoder.fit(df)
        df = self.encoder.transform(df)

        # 🔥 FIT TEXT
        self.text.fit(df)

        return self

    def transform(self, df):

        df = clean_data(df)
        df = create_features(df)
        df = add_geo(df)

        df = self.feature_engineer.transform(df)
        df = self.encoder.transform(df)
        df = self.text.transform(df)

        return df