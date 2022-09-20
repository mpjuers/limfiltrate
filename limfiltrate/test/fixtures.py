#!/usr/bin/env python3
# Copyright 2022 Mark Juers

import os
import random

from dash import dash_table
import pandas as pd
import pytest as pt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from limfiltrate.src.classes import Analysis

random.seed(666)
dirname = os.path.dirname(__file__)
data_path = os.path.join(
    dirname, "../Data/2022-07-26_test4_with-classification.csv"
)


@pt.fixture
def data():
    data = pd.read_csv(data_path)
    data.columns = data.columns = (
        data.columns.str.replace(" ", "_")
        .str.replace("/", "-")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.lower()
    )
    classes = data["class"]
    filtered_data = data.set_index("capture_id").select_dtypes(
        include=["float64"]
    )
    filtered_data["class"] = classes
    return filtered_data


@pt.fixture
def analysis():
    return Analysis(data_path)


@pt.fixture
def pcafixture(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.drop("class", axis=1))
    pca = PCA()
    return pd.DataFrame(
        pca.fit_transform(scaled),
        columns=["PC" + str(i) for i in range(1, scaled.shape[1] + 1)],
    )


@pt.fixture
def customdata(data):
    n_sample = 20
    customdata = random.sample(list(data.index), n_sample)
    return customdata


@pt.fixture
def summary(data):
    return (
        (
            data.drop("class", axis=1)
            .melt()
            .groupby("variable")
            .agg(["min", "max", "mean", "std"])
        )
        .droplevel(0, axis=1)
        .round(2)
    )


@pt.fixture
def data_table(summary):
    summary.insert(0, "particle_property", summary.index)
    table = dash_table.DataTable(
        summary.round(2).to_dict("records"),
        [{"name": i, "id": i} for i in summary.columns],
    )
    return table
