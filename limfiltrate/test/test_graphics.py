#!/usr/bin/env python3
# Copyright 2022 Mark Juers

import os
from pickle import dumps

import plotly.graph_objs as go
from dash import dash_table

from limfiltrate.test.fixtures import (
    data,
    customdata,
    summary,
    analysis,
    data_table,
)
from limfiltrate.src.classes import Graphics


dirname = os.path.dirname(__file__)
data_path = os.path.join(
    dirname, "../Data/2022-07-26_test4_with-classification.csv"
)


def test_generate_pca_plot(analysis):
    df = analysis.generate_pca()["transformed_data"].iloc[:, 0:5]
    dimensions = [
        {"label": i, "values": tuple(value)} for i, value in df.items()
    ]
    fig = go.Figure(
        go.Splom(
            dimensions=dimensions,
            showlowerhalf=False,
        )
    )
    assert dumps(fig) == dumps(Graphics(analysis).generate_pca_plot())


def test_generate_data_table(analysis, data_table):
    table = Graphics(analysis).generate_data_table()
    assert dumps(table) == dumps(data_table)
