#!/usr/bin/env python3

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

pca = PCA()
scaler = MinMaxScaler()

df = pd.read_csv("Data/2022-07-26_test4_with-classification.csv")

df_scaled = scaler.fit_transform(df.select_dtypes(np.number))
pca_out = pca.fit_transform(df_scaled)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

n_pcs = 5

fig = px.scatter_matrix(
    pca_out,
    labels=labels,
    dimensions=range(n_pcs),
    color=df["class"],
)

fig.update_traces(diagonal_visible=False)
# fig.show()

fig2 = px.imshow(pca.components_)
fig2.show()
