#!/usr/bin/env python3
# Copyright 2022 Neko Juers

import json
import os
import random

from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Analysis:
    def __init__(self, csv):
        """
        csv (string): Path to the file containing the data for analysis.

        returns: None

        Attributes:
            data (DataFrame): The original dataframe, appropriately formatted.
            data_filtered (DataFrame): The dataframe refined by filter.
        """
        self.data = pd.read_csv(csv)
        self.data.columns = (
            self.data.columns.str.replace(" ", "_")
            .str.replace("/", "-")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.lower()
        )
        try:
            self.classes = self.data["class"]
        except KeyError:
            self.classes = ["" for row in range(self.data.shape[0])]
        self.data = self.data.set_index("capture_id").select_dtypes(
            include=["float64"]
        )
        self.data_filtered = self.data
        return None

    def filter_data(self, customdata, drop_columns=None):
        """
        customdata (list-like): a list of indices to retain in analysis and plotting step

        returns: Analysis
            The Analysis object with the data_filtered updated
        """
        if drop_columns is not None:
            self.data_filtered = self.data.drop(drop_columns, axis=1).loc[
                customdata
            ]
        else:
            self.data_filtered = self.data.loc[customdata]
        return self

    def scale(self):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.data_filtered)
        return pd.DataFrame(scaled, columns=self.data.columns)

    def generate_pca(self, *args, **kwargs):
        """
        **kwargs:
            customdata (list-like): See documentation for self.filter_data()

        returns: dict
            pca (PCA): the fit pca object
            transformed_data (DataFrame): data transformed according to pca
        """
        scaled = self.scale()
        pca = PCA()
        df = pd.DataFrame(
            pca.fit_transform(scaled),
            columns=["PC" + str(i) for i in range(1, scaled.shape[1] + 1)],
        )
        return {"pca": pca, "transformed_data": df}

    def summarize(
        self, stats=["min", "max", "mean", "std"], precision=5, *args, **kwargs
    ):
        df = (
            (self.data_filtered.melt().groupby("variable").agg(stats))
            .droplevel(0, axis=1)
            .round(precision)
        )
        return df


class Graphics:
    def __init__(self, analysis):
        """
        analysis (Analysis): An analysis instance containing data and operations.
        """
        self.analysis = analysis
        self.data = analysis.data
        return None

    def generate_pca_plot(self, pcs_to_show=range(0, 5)):
        """
        Generate Splom for PCs selected.

        pcs_to_show (list-like): A sequence of numbers representing which
            principle components to show in the Splom.
        """
        pca = self.analysis.generate_pca()
        df = pca["transformed_data"].iloc[:, pcs_to_show]
        # print(df.head())
        dimensions = [
            {"label": column, "values": tuple(value)}
            for column, value in df.items()
        ]
        fig = go.Figure(
            go.Splom(
                dimensions=dimensions,
                showlowerhalf=False,
                customdata=self.data.index,
            )
        )
        return fig

    def generate_data_table(self, precision=2):
        """
        Generate and style data table.
        """
        summary = self.analysis.summarize()
        summary.insert(0, "particle_property", summary.index)
        scaled = self.analysis.data_filtered.to_numpy()
        data_mean = scaled.mean()
        feature_mean = scaled.mean(axis=0)
        ss_total = ((data_mean - scaled) ** 2).sum()
        ss_between = (feature_mean - data_mean) ** 2
        ss_explained = ss_between / ss_total

        summary.insert(len(summary.columns), "ss_explained", ss_explained)
        summary = summary.sort_values("ss_explained", ascending=False)
        padded = pd.DataFrame(
            np.pad(
                [
                    self.analysis.data_filtered.shape[0],
                ],
                (0, summary.shape[1] - 1),
            ).reshape(1, -1),
            columns=summary.columns,
        )
        summary = summary.append(padded)
        table = dash_table.DataTable(
            # Data table takes data frame as dict.
            summary.to_dict("records"),
            [{"name": i, "id": i} for i in summary.columns],
            style_as_list_view=True,  # Remove vertical cell lines.
            # Align all text to left except particle property.
            style_cell={"textAlign": "left"},
            style_cell_conditional=[
                {
                    "if": {"column_id": "particle_property"},
                    "textAlign": "right",
                }
            ],
        )
        return table


if __name__ == "__main__":

    # Set path to script directory.
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    datapath = "../Data/2022-09-30.csv"
    analysis = Analysis(datapath)
    graphics = Graphics(analysis)
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    table = graphics.generate_data_table()
    fig = graphics.generate_pca_plot(range(1, 5))

    @app.callback(
        Output("pcaPlot", "figure"),
        Input("pcsToDisplay", "value"),
        Input("pcaRecalc", "n_clicks"),
        State("pcaPlot", "selectedData"),
    )
    def _pca_plot(value, n_clicks, selectedData):
        try:
            points = [point["customdata"] for point in selectedData["points"]]
        except TypeError:
            points = analysis.data.index
        graphics.analysis.filter_data(points)
        graphics.analysis.generate_pca()
        return graphics.generate_pca_plot(range(value[0], value[1] + 1))

    @app.callback(
        Output("dataTable", "children"), Input("pcaPlot", "selectedData")
    )
    def _data_table(selectedData):
        # Exception handling for no data selected: populate with all data.
        try:
            points = [point["customdata"] for point in selectedData["points"]]
        except TypeError:
            points = analysis.data.index
        graphics.analysis.filter_data(points)
        table = graphics.generate_data_table()
        return table

    app.layout = html.Div(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.RangeSlider(
                                    0,
                                    10,
                                    step=1,
                                    value=[0, 4],
                                    id="pcsToDisplay",
                                    marks={
                                        i: str(i + 1) for i in range(0, 10)
                                    },
                                ),
                                dcc.Graph(
                                    style={"width": "90vh", "height": "90vh"},
                                    id="pcaPlot",
                                ),
                                html.Button("Recalculate PCA", id="pcaRecalc"),
                            ]
                        ),
                    ],
                    width=7,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id="dataTable",
                        ),
                    ],
                    width=5,
                ),
            ],
        ),
    )

    app.run_server(debug=True)
