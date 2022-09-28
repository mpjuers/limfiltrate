#!/usr/bin/env python3
# Copyright 2022 Mark Juers

import os
import random
from pickle import dumps

import plotly.graph_objs as go
import pytest
from dash import dash_table

from limfiltrate.src.classes import Graphics
from limfiltrate.test.fixtures import *


dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, "../Data/2022-07-26_test4_with-classification.csv")


@pytest.mark.parametrize(
    "range_in,expected", [(range(0, 5), "pca_plot_0_5"), (range(1, 3), "pca_plot_1_3")]
)
def test_generate_pca_plot(request, analysis, range_in, expected):
    fig = request.getfixturevalue(expected)
    print(fig)
    assert fig == Graphics(analysis).generate_pca_plot(range_in)


def test_generate_data_table(analysis, data_table):
    table = Graphics(analysis).generate_data_table()
    assert dumps(table) == dumps(data_table)


def test_random_seed(analysis):
    random.seed(42)
    pca0 = Graphics(analysis).generate_pca_plot()
    random.seed(666)
    pca1 = Graphics(analysis).generate_pca_plot()
    assert pca0 == pca1
