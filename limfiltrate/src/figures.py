#!/usr/bin/env python3
# Copyright 2022 Neko Juers

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_scatter(
    x,
    y,
    components,
    show_biplot=True,
    biplot_scaling=20,
    biplot_labels=None,
    *args,
    **kwargs
):
    fig = px.scatter(x=x, y=y)
    components = components * biplot_scaling
    if show_biplot is True:
        for i in range(components.shape[0]):
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=components[i, 0],
                y1=components[i, 1],
                hover_data=biplot_labels[i] if biplot_labels is not None else None,
            )
    return fig


if __name__ == "__main__":
    df = pd.read_csv("../../Data/2022-09-30.csv")
    df = df.select_dtypes(np.number)
    scale = StandardScaler()
    scaled = scale.fit_transform(df)
    pca = PCA()
    pca_data = pd.DataFrame(pca.fit_transform(scaled))
    components_T = pca.components_.T
    fig = generate_scatter(
        pca_data.iloc[:, 5],
        pca_data.iloc[:, 7],
        np.stack(
            [
                components_T[:, 5].reshape(1, -1),
                components_T[:, 7].reshape(1, -1),
            ],
            axis=1,
        )[0].T,
        biplot_labels=df.columns,
    )
    fig.show()
