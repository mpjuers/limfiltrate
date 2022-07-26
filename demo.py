#!/usr/bin/env python3

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

pca = PCA()
scaler = MinMaxScaler()

df = pd.read_csv("Data/2022-07-26_test4.csv")
df = df.select_dtypes(np.number)

df_scaled = scaler.fit_transform(df)
components = pca.fit_transform(df_scaled)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

n_pcs = 5

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(n_pcs),
)

fig.update_traces(diagonal_visible=False)
fig.show()
