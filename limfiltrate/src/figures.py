#!/usr/bin/env python3
# Copyright 2022 Neko Juers

import itertools as it

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_scatter(
    x, y, components, show_biplot=True, biplot_scaling=20, *args, **kwargs
):
    fig = px.scatter(x=x, y=y)
    components = components * biplot_scaling
    if show_biplot is True:
        for i in range(components.shape[0]):
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=components.iloc[i, 0],
                y1=components.iloc[i, 1],
            )
        fig.add_trace(
            go.Scatter(
                x=components.iloc[:, 0],
                y=components.iloc[:, 1],
                hovertext=components.index,
                mode="markers",
            )
        )
        fig.update_layout(xaxis_title=None, yaxis_title=None)
    return fig


if __name__ == "__main__":
    df = pd.read_csv("../../Data/2022-09-30.csv")
    df = df.select_dtypes(np.number).drop(
        ["Group ID", "Source Image", "Elapsed Time"]
        + [x + " " + y for x, y in it.product(["Ch1", "Ch2"], ["Area", "Width"])]
        + ["Capture" + " " + x for x in ["X", "Y", "ID"]],
        axis=1,
    )
    scale = StandardScaler()
    scaled = scale.fit_transform(df)
    pca = PCA()
    pca_data = pd.DataFrame(pca.fit_transform(scaled))
    components_T = pd.DataFrame(pca.components_.T, index=df.columns)
    fig = generate_scatter(
        pca_data.iloc[:, 1],
        pca_data.iloc[:, 2],
        pd.concat(
            [
                components_T.iloc[:, 1],
                components_T.iloc[:, 2],
            ],
            axis=1,
        ),
        biplot_scaling=30,
    )
    fig.show()
