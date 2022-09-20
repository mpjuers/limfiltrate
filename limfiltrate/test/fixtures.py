#!/usr/bin/env python3
# Copyright 2022 Mark Juers

import os
import random

import pandas as pd
import pytest as pt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
def pcafixture(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.drop("class", axis=1))
    pca = PCA()
    return pca.fit_transform(scaled)


@pt.fixture
def customdata(data):
    n_sample = 20
    customdata = random.sample(list(data.index), n_sample)
    return customdata


@pt.fixture
def summary(data):
    return (
        data.drop("class", axis=1)
        .melt()
        .groupby("variable")
        .agg(["min", "max", "mean", "std"])
    )
