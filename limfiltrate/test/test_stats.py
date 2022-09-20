#!/usr/bin/env python3
# Copyright 2022 Mark Juers

import os
import pandas as pd
from pickle import dumps

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from limfiltrate.test.fixtures import data, pcafixture
from limfiltrate.src.classes import Analysis

dirname = os.path.dirname(__file__)
data_path = os.path.join(
    dirname, "../Data/2022-07-26_test4_with-classification.csv"
)


def test_pca(pcafixture, data):
    analysis = Analysis(data_path).generate_pca()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.drop("class", axis=1))
    pca = PCA()
    pca.fit_transform(scaled)
    df = pd.DataFrame(
        pca.fit_transform(scaled), columns=scaler.feature_names_in_
    )
    assert dumps(pcafixture) == dumps(analysis["transformed_data"])
