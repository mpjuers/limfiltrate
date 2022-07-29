#!/usr/bin/env python3

import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

pca = PCA()
scaler = MinMaxScaler()

df = pd.read_csv("Data/2022-07-28_autoimage-test_2.csv")
df.columns = (
    df.columns.str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("/", "-")
    .str.replace(".", "")
    .str.lower()
)

df["longskinnies"] = df.aspect_ratio.map(
    lambda x: "longskinny" if x < 0.25 else "round"
)

df_scaled = scaler.fit_transform(df.select_dtypes(np.number))
pca_out = pca.fit_transform(df_scaled)

labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

n_pcs = range(2, 7)

fig = px.scatter_matrix(
    pca_out,
    labels=labels,
    dimensions=n_pcs,
    color=df["longskinnies"],
    opacity=0.15,
)

fig.update_traces(diagonal_visible=False)

app.layout = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        dcc.Graph(figure=fig),
                    ),
                ],
                width=9,
            ),
            dbc.Col(
                [
                    html.Div(
                        "filler text",
                    ),
                ],
                width=3,
            ),
        ],
    ),
)


if __name__ == "__main__":
    app.run_server(debug=True)
