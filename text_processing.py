import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class TextProcessor:

    def __init__(self):

        # 🔒 EXACT SAME AS YOUR NOTEBOOK
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=5,
            max_df=0.90,
            stop_words='english'
        )

        self.text_cols = [
            'job_description',
            'NEW_NAR_WHAT_HAPPENED',
            'NEW_NAR_BEFORE_INCIDENT',
            'NEW_INCIDENT_LOCATION',
            'NEW_NAR_INJURY_ILLNESS',
            'NEW_NAR_OBJECT_SUBSTANCE',
            'NEW_INCIDENT_DESCRIPTION'
        ]

    # -------------------------
    # FIT
    # -------------------------
    def fit(self, df):

        df = df.copy()

        # combine text
        df['combined_text'] = df[self.text_cols].fillna('').astype(str).agg(' '.join, axis=1)

        # cleaning (same as notebook)
        df['combined_text'] = df['combined_text'].str.lower()
        df['combined_text'] = df['combined_text'].apply(
            lambda x: re.sub(r'[^a-z\s]', '', x)
        )

        # fit TF-IDF
        self.vectorizer.fit(df['combined_text'])

        return self

    # -------------------------
    # TRANSFORM
    # -------------------------
    def transform(self, df):

        df = df.copy()

        df['combined_text'] = df[self.text_cols].fillna('').astype(str).agg(' '.join, axis=1)

        df['combined_text'] = df['combined_text'].str.lower()
        df['combined_text'] = df['combined_text'].apply(
            lambda x: re.sub(r'[^a-z\s]', '', x)
        )

        tfidf_matrix = self.vectorizer.transform(df['combined_text'])

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

        # reset index (VERY IMPORTANT — same as notebook)
        df = df.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)

        # merge
        df = pd.concat([df, tfidf_df], axis=1)

        # drop original text
        df = df.drop(columns=self.text_cols + ['combined_text'])

        return df