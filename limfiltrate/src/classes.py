#!/usr/bin/env python3

import pandas as pd


class Analysis:
    def __init__(self, csv):
        """
        csv (string): Path to the file containing the data for analysis.
        """
        self.data = pd.read_csv(csv)
        self.data.columns = (
            self.data.columns.str.replace(" ", "_")
            .str.replace("/", "-")
            .str.replace("(", "--")
            .str.replace(")", "--")
            .str.lower()
        )
        print(self.data.columns)
        self.data = self.data.set_index("capture_id")
        return None

    def filter_data(self, drop_columns=None):
        if drop_columns is not None:
            print(self.data.columns)
            return self.data.drop(drop_columns, axis=1)
        else:
            return self.data


if __name__ == "__main__":
    print(Analysis("../Data/2022-07-26_test4_with-classification.csv").data)
