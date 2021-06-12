import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_uploader as du
import dash_daq as daq

from server import app

from views.data_callbks import *

data_page = html.Div([

    dbc.Container([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([html.Label("Estimated Scene Information", style={"color": "#edefeb"})], className='box',
                             style={'padding': '0px', "padding-top": "6px", "padding-bottom": "2px",
                                    "text-align": "center",
                                    "background-color": "#4b9072", 'border-radius': '15px', 'box-shadow': '0 0 0',
                                    'box-border': 0}),
                    html.Div([
                        dcc.Graph(id="graph-input", config=data_graph_config,
                                  figure=get_data_graph_figure(osp.join(PATH_IN, 'im.png')))
                    ], className='box', id="html-graph-input"),
                ], style={"width": "33.3%"}),
                html.Div([
                    html.Div([
                        dcc.Graph(id="graph-light", config=data_graph_config,
                                  figure=get_envmap_figure(osp.join(PATH_IN, 'envmap.png.npz')))
                    ], className='box', id="html-graph-light"),
                ], style={"width": "66.6%"}),
            ], className='row'),
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id="graph-normal", config=data_graph_config,
                                  figure=get_data_graph_figure(osp.join(PATH_IN, 'normal.png')))
                    ], className='box', id="html-graph-normal"),
                ], style={"width": "33.3%"}),
                html.Div([
                    html.Div([
                        dcc.Graph(id="graph-rough", config=data_graph_config,
                                  figure=get_data_graph_figure(osp.join(PATH_IN, 'rough.png'), True))
                    ], className='box', id="html-graph-rough"),
                ], style={"width": "33.3%"}),
                html.Div([
                    html.Div([
                        dcc.Graph(id="graph-albedo", config=data_graph_config,
                                  figure=get_data_graph_figure(osp.join(PATH_IN, 'albedo.png')))
                    ], className='box', id="html-graph-albedo"),
                ], style={"width": "33.3%"}),
            ], className="row")
        ], id="Viewer-of-estimate"),
        html.Br(),
        # html.Hr(style={'borderColor': '#4b9072'}),
        html.Hr(),
        html.Br(),
        html.Div([
            html.Div([
                html.Div([
                    html.Div([html.Label("Intermediate Files in Rendering Process", style={"color": "#edefeb"})], className='box',
                             style={'padding': '0px', "padding-top": "6px", "padding-bottom": "2px",
                                    "text-align": "center",
                                    "background-color": "#4b9072", 'border-radius': '15px', 'box-shadow': '0 0 0',
                                    'box-border': 0}),
                    html.Div([
                        dcc.Graph(id="graph-output", config=data_graph_config,
                                  figure=get_data_graph_figure(osp.join(PATH_OUT, 'im.png'), _height=500))
                    ], className='box', id="html-graph-output"),
                ], style={"width": "50%"}),
                html.Div([
                    html.Div([
                        dcc.Graph(id="graph-scene-rgbe", config=data_graph_config,
                                  figure=get_rgbe_figure(osp.join(PATH_OUT, 'scene_1.rgbe')))
                    ], className='box', id="html-graph-scene-rgbe"),
                    html.Div([
                        dcc.Graph(id="graph-obj-mask", config=data_graph_config,
                                  figure=get_data_graph_figure(osp.join(PATH_OUT, 'scene_objmask_1.png'),
                                                               _height=245))
                    ], className='box', id="html-graph-obj-mask"),
                ], style={"width": "25%"}),
                html.Div([
                    html.Div([
                        dcc.Graph(id="graph-bkg-rgbe", config=data_graph_config,
                                  figure=get_rgbe_figure(osp.join(PATH_OUT, 'scene_bkg_1.rgbe')))
                    ], className='box', id="html-graph-bkg-rgbe"),
                    html.Div([
                        dcc.Graph(id="graph-scene-mask", config=data_graph_config,
                                  figure=get_data_graph_figure(osp.join(PATH_OUT, 'scenemask_1.png'),
                                                               _height=245))
                    ], className='box', id="html-graph-scene-mask"),
                ], style={"width": "25%"}),

            ], className='row')
        ], id="Viewer-of-render"),
        html.Br(),
        html.Hr(),
    ], fluid=True, style={'width': '95%', 'margin-top': '5px'}),

    html.Div([
        make_data_tooltip('Origin Image of Scene', 'html-graph-input', 'bottom'),
        make_data_tooltip('Lighting (SG)', 'html-graph-light', 'bottom'),
        make_data_tooltip('Normal (vector)', 'html-graph-normal', 'bottom'),
        make_data_tooltip('Roughness (number)', 'html-graph-rough', 'bottom'),
        make_data_tooltip('Albedo (vector)', 'html-graph-albedo', 'bottom'),
        make_data_tooltip('Final Mixed Output Image', 'html-graph-output', 'bottom'),
        make_data_tooltip('Rendered Scene (RGBE)', 'html-graph-scene-rgbe', 'bottom'),
        make_data_tooltip('Mask of Object (PNG)', 'html-graph-obj-mask', 'bottom'),
        make_data_tooltip('Rendered Plane (RGBE)', 'html-graph-bkg-rgbe', 'bottom'),
        make_data_tooltip('Mask of Plane (PNG)', 'html-graph-scene-mask', 'bottom'),
    ])
])
