#!/usr/bin/env python3

from limfiltrate.src.classes import Analysis

def test_analysis_init():
    assert Analysis("Data/2022-07-26_test4_with-classification.csv")

def test_read_data():
    assert Analysis("Data/2022-07-26_test4_with-classification.csv").read_data()
