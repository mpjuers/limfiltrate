#!/usr/bin/env python3

import os
from pickle import dumps

import pandas as pd
from limfiltrate.src.classes import Analysis

dirname = os.path.dirname(__file__)
data_path = os.path.join(
    dirname, "../Data/2022-07-26_test4_with-classification.csv"
)
data = pd.read_csv(data_path)

data.columns = data.columns = (
    data.columns.str.replace(" ", "_")
    .str.replace("/", "-")
    .str.replace("(", "--")
    .str.replace(")", "--")
    .str.lower()
)
filtered_data = data.set_index("capture_id")


def test_analysis_init():
    assert Analysis(data_path)


def test_filter_data():
    assert dumps(Analysis(data_path).filter_data()) == dumps(
        filtered_data
    )
