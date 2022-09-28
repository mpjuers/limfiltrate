#!/usr/bin/env python3
# Copyright 2022 Mark Juers

import os
from pickle import dumps

from limfiltrate.test.fixtures import data, customdata, summary
from limfiltrate.src.classes import Analysis


dirname = os.path.dirname(__file__)
data_path = os.path.join(
    dirname, "../Data/2022-07-26_test4_with-classification.csv"
)


def test_analysis_init():
    assert Analysis(data_path)


def test_filter_data(data, customdata):
    assert dumps(
        Analysis(data_path).filter_data(customdata).data_filtered
    ) == dumps(data.loc[customdata])


def test_summarize(data, summary):
    assert dumps(summary) == dumps(Analysis(data_path).summarize())
