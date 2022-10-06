#!/usr/bin/env python3
# Copyright 2022 Neko Juers

import os
import random

from dash import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest as pt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from limfiltrate.src.classes import Analysis

random.seed(666)

dirname = os.path.dirname(__file__)
data_path = os.path.join(
    dirname, "../../Data/2022-07-26_test4_with-classification.csv"
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
    return filtered_data


@pt.fixture
def analysis():
    return Analysis(data_path)


@pt.fixture
def pcafixture(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA()
    return pd.DataFrame(
        pca.fit_transform(scaled),
        columns=["PC" + str(i) for i in range(1, scaled.shape[1] + 1)],
    )


@pt.fixture
def pcafixture_analysis(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA()
    return pca.fit(data)


@pt.fixture
def customdata(data):
    n_sample = 20
    customdata = random.sample(list(data.index), n_sample)
    return customdata


@pt.fixture
def summary(data):
    return (
        (data.melt().groupby("variable").agg(["min", "max", "mean", "std"]))
        .droplevel(0, axis=1)
        .round(5)
    )


@pt.fixture
def data_table(summary, data):
    data = data.to_numpy()
    summary.insert(0, "particle_property", summary.index)
    data_mean = data.mean()
    feature_mean = data.mean(axis=0)
    ss_total = ((data_mean - data) ** 2).sum()
    ss_between = (feature_mean - data_mean) ** 2
    ss_explained = ss_between / ss_total
    summary.insert(len(summary.columns), "ss_explained", ss_explained)
    summary = summary.sort_values("ss_explained", ascending=False)
    padded = pd.DataFrame(
        np.pad(
            [
                data.shape[0],
            ],
            (0, summary.shape[1] - 1),
        ).reshape(1, -1),
        columns=summary.columns,
    )
    summary = summary.append(padded)
    table = dash_table.DataTable(
        summary.to_dict("records"),
        [{"name": i, "id": i} for i in summary.columns],
        style_as_list_view=True,
        style_cell={"textAlign": "left"},
        style_cell_conditional=[
            {
                "if": {"column_id": "particle_property"},
                "textAlign": "right",
            }
        ],
    )
    return table


@pt.fixture
def pca_plot_0_5(analysis, data, pcafixture, pcafixture_analysis):
    df = pcafixture.iloc[:, range(0, 5)]
    pca = analysis.generate_pca()["pca"]
    dimensions = [
        {
            "label": column + "<br>{:.3f}".format(var_percent),
            "values": tuple(value),
        }
        for (
            column,
            value,
        ), var_percent in zip(df.items(), pca.explained_variance_ratio_)
    ]
    fig = go.Figure(
        go.Splom(
            dimensions=dimensions,
            showlowerhalf=False,
            customdata=data.index,
        )
    )
    return fig


@pt.fixture
def pca_plot_1_3(analysis, data):
    df = analysis.generate_pca()["transformed_data"].iloc[:, range(1, 3)]
    pca = analysis.generate_pca()["pca"]
    dimensions = [
        {
            "label": column + "<br>{:.3f}".format(var_percent),
            "values": tuple(value),
        }
        for (
            column,
            value,
        ), var_percent in zip(df.items(), pca.explained_variance_ratio_)
    ]
    fig = go.Figure(
        go.Splom(
            dimensions=dimensions,
            showlowerhalf=False,
            customdata=data.index,
        )
    )
    return fig
