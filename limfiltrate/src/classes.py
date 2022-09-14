#!/usr/bin/env python3

import pandas as pd

class Analysis:
    def __init__(self, csv):
        self.csv = csv
        return None 

    def read_data(self):
        pd.read_csv(self.csv)
        return True
