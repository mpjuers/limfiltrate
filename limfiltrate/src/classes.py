#!/usr/bin/env python3

import pandas as pd

class Analysis:
    def __init__(self, csv):
        self.data = pd.read_csv(csv)
        return None 

