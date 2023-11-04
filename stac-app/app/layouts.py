import datetime
from datetime import date

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash_extensions.enrich import dcc, html

from app.config import init_bounds

navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Img(
                                    src="https://earthdaily.com/wp-content/themes/earthdaily-2023/src/assets/images/logo-1.svg",
                                    height="150px",
                                )
                            ],
                            style={"margin-left": "10px"},
                        )
                    ),
                ],
                align="center",
            )
        ),
    ],
    color="#fff",
)


def get_main_page() -> html.Div:
    """
    Defines the layout for the main page
    Returns:
        html.Div
    """
    today = datetime.datetime.now()

    div = html.Div(
        [
            dcc.Store(id="dataset-store", storage_type="local"),
            dcc.Store(id="dates-store", storage_type="local"),
            dcc.Store(id="color-bounds", storage_type="local"),
            dcc.Store(id="color-baseline-url", storage_type="local"),
            dcc.Store(id="color-comparison-url", storage_type="local"),
            dcc.Store(id="segmentation-bounds", storage_type="local"),
            dcc.Store(id="segmentation-baseline-url", storage_type="local"),
            dcc.Store(id="segmentation-comparison-url", storage_type="local"),
            html.Div([], id="init-hook"),
            navbar,
            dbc.CardGroup(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "EDS Collection", style={"font-weight": "bold"}
                                ),
                                dcc.Dropdown(
                                    id="eds-collection-selector",
                                    options=[
                                        {
                                            "label": "Sentinel 2 L2A",
                                            "value": "sentinel-2-l2a",
                                        },
                                        {"label": "Venus L2A", "value": "venus-l2a"},
                                    ],
                                ),
                            ]
                        )
                    ),
                    dbc.Card(
                        dbc.CardBody([
                            html.H5('Scene Cloud Cover Percentage', style={'font-weight': 'bold'}),
                            dcc.Slider(0, 100, 1, value=10, marks=None, id='cloud-cover-threshold',
                                    tooltip={"placement": "bottom", "always_visible": True})

                        ])),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Date Range", style={"font-weight": "bold"}),
                                html.Div(
                                    [
                                        dcc.DatePickerRange(
                                            id="date-picker-range",
                                            min_date_allowed=date(2017, 1, 1),
                                            max_date_allowed=date(
                                                today.year, today.month, today.day
                                            ),
                                            initial_visible_month=date(2020, 7, 1),
                                            start_date=date(2020, 7, 1),
                                            end_date=date(2020, 8, 31),
                                        ),
                                        html.Div(
                                            id="output-container-date-picker-range"
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Button("Search", id="search-btn"),
                                html.Br(),
                                html.Br(),
                                dbc.Spinner(html.Div(id="loading-data-cube")),
                            ]
                        )
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Baseline Date", style={"font-weight": "bold"}),
                                dcc.Dropdown(id="date-selector-baseline", options=[]),
                            ]
                        )
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Comparison Date", style={"font-weight": "bold"}
                                ),
                                dcc.Dropdown(id="date-selector-comparison", options=[]),
                                html.Br(),
                                dbc.Spinner(html.Div(id="loading-color")),
                            ]
                        )
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5(
                                    "Unsupervised Segmentation", style={"font-weight": "bold"}
                                ),
                                dcc.Dropdown(
                                    id="n-classes-selector",
                                    options=[
                                        {
                                            "label": "nclasses=3",
                                            "value": 3,
                                        },
                                        {"label": "nclasses=4", "value": 4},
                                        {"label": "nclasses=5", "value": 5},
                                    ],
                                ),
                                html.Br(),
                                dbc.Button("Compute", id="segmentation-btn"),
                                html.Br(),
                                dbc.Spinner(html.Div(id="loading-segmentation")),


                            ]
                        )
                    ),
                ],
                style={
                    "width": "15%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            ),
            dbc.CardGroup(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Spinner(html.Div(id="loading-render")),
                                html.Br(),
                                dl.Map(
                                    [
                                        dl.Pane(
                                            dl.LayersControl(id="raster-layers"),
                                            style={"zIndex": 1},
                                        ),
                                        dl.Pane(
                                            dl.LayersControl(id="vector-layers"),
                                            style={"zIndex": 2},
                                        ),
                                        dl.Pane(
                                            dl.FeatureGroup(
                                                [dl.EditControl(id="edit-control")]
                                            ),
                                            style={"zIndex": 3},
                                        ),
                                    ],
                                    id="map",
                                    bounds=init_bounds,
                                    style={"height": "70vh", "margin": "auto"},
                                ),
                            ]
                        )
                    )
                ],
                style={
                    "width": "85%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            ),
        ]
    )

    return div
