#!/usr/bin/env python3
# Copyright 2022 Neko Juers

import os
import random

from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc
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
        classes = self.data["class"]
        self.data = self.data.set_index("capture_id").select_dtypes(
            include=["float64"]
        )
        self.data["class"] = classes
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

    def generate_pca(self, *args, **kwargs):
        """
        **kwargs:
            customdata (list-like): See documentation for self.filter_data()

        returns: dict
            pca (PCA): the fit pca object
            transformed_data (DataFrame): data transformed according to pca
        """
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.data.drop("class", axis=1))
        pca = PCA()
        df = pd.DataFrame(
            pca.fit_transform(scaled),
            columns=["PC" + str(i) for i in range(1, scaled.shape[1] + 1)],
        )
        return {"pca": pca, "transformed_data": df}

    def summarize(
        self, stats=["min", "max", "mean", "std"], precision=2, *args, **kwargs
    ):
        return (
            (
                self.data.drop("class", axis=1)
                .melt()
                .groupby("variable")
                .agg(stats)
            )
            .droplevel(0, axis=1)
            .round(precision)
        )


class Graphics:
    def __init__(self, analysis):
        self.analysis = analysis
        self.data = analysis.data
        return None

    def generate_pca_plot(self, pcs_to_show=range(0, 5)):
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
                diagonal={"visible": False},
            )
        )
        return fig

    def generate_data_table(self, precision=2):
        summary = self.analysis.summarize()
        summary.insert(0, "particle_property", summary.index)
        summary = summary.round(precision)
        table = dash_table.DataTable(
            summary.to_dict("records"),
            [{"name": i, "id": i} for i in summary.columns],
            style_as_list_view=True,  # Remove vertical cell lines.
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

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    datapath = "../Data/2022-07-26_test4_with-classification.csv"
    analysis = Analysis(datapath)
    graphics = Graphics(analysis)
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    table = graphics.generate_data_table()
    fig = graphics.generate_pca_plot(range(1, 5))

    @app.callback(Output("pcaPlot", "figure"), Input("pcsToDisplay", "value"))
    def _pca_plot(value):
        return graphics.generate_pca_plot(range(*value))

    app.layout = html.Div(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1("PCs to display"),
                                dcc.RangeSlider(
                                    0,
                                    10,
                                    step=1,
                                    value=[1, 6],
                                    id="pcsToDisplay",
                                    marks={
                                        i: str(i + 1) for i in range(0, 10)
                                    },
                                ),
                                dcc.Graph(
                                    style={"width": "90vh", "height": "90vh"},
                                    id="pcaPlot",
                                ),
                            ]
                        ),
                    ],
                    width=9,
                ),
                dbc.Col(
                    [
                        html.Div(
                            table,
                        ),
                    ],
                    width=3,
                ),
            ],
        ),
    )

    app.run_server(debug=True)
