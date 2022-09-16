#!/usr/bin/env python3

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Analysis:
    def __init__(self, csv):
        """
        csv (string): Path to the file containing the data for analysis.
        """
        self.data = pd.read_csv(csv)
        self.data.columns = (
            self.data.columns.str.replace(" ", "_")
            .str.replace("/", "-")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.lower()
        )
        classes = self.data["class"]
        self.data = self.data.set_index("capture_id").select_dtypes(
            include=["float64"]
        )
        self.data["class"] = classes
        return None

    def filter_data(self, customdata, drop_columns=None):
        """
        customdata (list-like): a list of indices to retain in analysis and plotting step
        """
        if drop_columns is not None:
            return self.data.drop(drop_columns, axis=1).loc[customdata]
        else:
            return self.data.loc[customdata]

    def generate_pca(self):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.data.drop("class", axis=1))
        pca = PCA()
        df = pca.fit_transform(scaled)
        return {"pca": pca, "transformed_data": df}


if __name__ == "__main__":

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print(
        Analysis(
            "../Data/2022-07-26_test4_with-classification.csv"
        ).generate_pca()
    )
