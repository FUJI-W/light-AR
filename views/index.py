import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_uploader as du
import dash_daq as daq

from views.index_callbks import *

index_page = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                width=3,
                children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                daq.Knob(
                                    id='knob-obj-size', style={"background": "transparent"}, size=70,
                                    # label="Object Size (%)", labelPosition="bottom",
                                    min=0, max=100, value=int(SCALE * 100),
                                    color={"gradient": True,
                                           "ranges": {"#e3f7d7": [0, 80], "#4b9072": [80, 100]}},
                                ),
                            ),
                            dbc.Col(
                                daq.LEDDisplay(
                                    id='knob-obj-size-output', color="#4b9072"
                                    # label="Object Size",
                                ),
                            )
                        ], align="center"),
                    ], className='box',
                        style={'width': '100%', 'margin-bottom': '20px',
                               'padding-top': '15px', 'padding-bottom': '15px'}),
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                daq.Knob(
                                    id='knob-obj-roughness', style={"background": "transparent"}, size=70,
                                    # label="Object Size (%)", labelPosition="bottom",
                                    min=0, max=100, value=int(ROUGHNESS * 100),
                                    color={"gradient": True,
                                           "ranges": {"#e3f7d7": [0, 80], "#4b9072": [80, 100]}},
                                ),
                            ),
                            dbc.Col(
                                daq.LEDDisplay(
                                    id='knob-obj-roughness-output', color="#4b9072",
                                    # label="Object Size",
                                ),
                            )
                        ], align="center"),
                    ], className='box',
                        style={'width': '100%', 'margin-bottom': '20px',
                               'padding-top': '15px', 'padding-bottom': '15px'}),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Div(
                                    dcc.Graph(id='color-picker', figure=get_color_picker_figure(),
                                              config={'displayModeBar': False, 'editable': False,
                                                      'scrollZoom': False}),
                                    style={'display': 'inline-block', 'width': '85%'}
                                ), ], width=8
                            ),
                            dbc.Col([
                                html.Div([
                                    html.Div(
                                        daq.LEDDisplay(id='color-R-output', value=int(COLOR_R * 255), size=16,
                                                       color="#d27656", ),
                                        style={'display': 'inline-block', 'width': '100%', }
                                    ),
                                    html.Div(
                                        daq.LEDDisplay(id='color-G-output', value=int(COLOR_G * 255), size=16,
                                                       color="#4b9072"),
                                        style={'display': 'inline-block', 'width': '100%', }
                                    ),
                                    html.Div(
                                        daq.LEDDisplay(id='color-B-output', value=int(COLOR_B * 255), size=16,
                                                       color="#293a66"),
                                        style={'display': 'inline-block', 'width': '100%', }
                                    ),
                                ])], width=4
                            ),
                        ], align='center'),
                        html.Div(id='color-picker-output')

                    ], className='box',
                        style={'width': '100%',
                               'padding-top': '0px', 'padding-bottom': '0px'}),
                ]
            ),
            dbc.Col(
                width=6,
                children=[
                    # html.Br(),
                    html.Div([
                        html.Div([
                            dcc.Tabs(
                                [
                                    dcc.Tab([dbc.Container(dcc.Graph(id='graph-picture'), id='div-graph-picture')],
                                            id="image-tabs",
                                            value="1",
                                            style={'padding': '0px',
                                                   'border': 0,
                                                   'border-radius': '5px',
                                                   'border-top-right-radius': '0px',
                                                   'border-bottom-right-radius': '0px',
                                                   'backgroundColor': '#edefeb'},
                                            selected_style={'padding': '0px',
                                                            'border': 0,
                                                            'border-radius': '5px',
                                                            'border-top-right-radius': '0px',
                                                            'border-bottom-right-radius': '0px',
                                                            'backgroundColor': '#4b9072'}
                                            ),
                                    dcc.Tab([html.Div([
                                        html.Video(
                                            controls=True,
                                            id='movie-player',
                                            src="inout/outputs/video.webm",
                                            autoPlay=True,
                                            width='100%'
                                        )],
                                        style={'height': '80%', 'margin': '15px', 'margin-top': '20px'}
                                    )],
                                        id="view-tabs",
                                        value="2",
                                        style={'padding': '0px',
                                               'border': 0,
                                               'border-radius': '0px',
                                               'backgroundColor': '#edefeb'},
                                        selected_style={'padding': '0px',
                                                        'border': 0,
                                                        'border-radius': '0px',
                                                        'backgroundColor': '#4b9072'}
                                    ),
                                    dcc.Tab([dbc.Container(dcc.Graph(id='graph-sunburst'), id='div-graph-sunburst')],
                                            id="sunburst-tabs",
                                            value="3",
                                            style={'padding': '0px',
                                                   'border': 0,
                                                   'border-radius': '5px',
                                                   'border-top-left-radius': '0px',
                                                   'border-bottom-left-radius': '0px',
                                                   'backgroundColor': '#edefeb'},
                                            selected_style={'padding': '0px',
                                                            'border': 0,
                                                            'border-radius': '5px',
                                                            'border-top-left-radius': '0px',
                                                            'border-bottom-left-radius': '0px',
                                                            'backgroundColor': '#4b9072'}
                                            ),
                                ],
                                # vertical=True,
                                id='tabs',
                                value='1',
                                style={
                                    'height': '5px',
                                    # 'width': '80%',
                                    'border': 0,
                                    'margin': '15px',
                                    'margin-top': '15px',
                                    'margin-bottom': '5px'
                                }
                            ),
                        ], style={'width': '100%', 'height': '100%'})
                    ], className='box',
                        style={'width': '100%', 'height': '80%',
                               'padding-top': '10px',
                               # 'padding-bottom': '15px',
                               'display': 'flex', 'align-items': 'center'}
                    ),
                    html.Div([
                        dbc.Container(
                            [
                                dbc.RadioItems(
                                    id="radios-system-mode",
                                    className="btn-group",
                                    labelClassName="radio-label",
                                    labelCheckedClassName="radio-checked",
                                    options=[
                                        {"label": "Memory Off", "value": 1},
                                        {"label": "Memory On", "value": 2},
                                        {"label": "Multi Obj", "value": 3},
                                    ],
                                    value=1,
                                ),
                            ],
                            className="radio-group"
                        )
                    ], id='radio-container', className='box',
                        style={'width': '100%', 'height': '13%', 'margin-top': '20px',
                               'padding-top': '0px', 'padding-bottom': '0px',
                               'display': 'flex', 'align-items': 'center'}
                    )
                ]
            ),
            dbc.Col(
                width=3,
                children=[
                    html.Div([
                        du.Upload(id='uploader', filetypes=['png'], upload_id='inputs',
                                  default_style={'height': '100%',
                                                 'minHeight': 1, 'lineHeight': 1,
                                                 'textAlign': 'center',
                                                 'outlineColor': '#ea8f32',
                                                 'font-family': 'Open Sans',
                                                 'font-size': '15px',
                                                 'font-weight': '500',
                                                 }),
                    ], className='box',
                        style={'width': '100%', 'height': '92px', 'margin-bottom': '20px',
                               'padding-top': '15px', 'padding-bottom': '15px'}
                    ),
                    html.Div(
                        dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Img(src=app.get_asset_url('obj_ball.png'), id='html-a-sphere',
                                                 height="80%", style={'margin': 'auto'}),
                                    ], id='html-sphere-block-color', className='box_objects'),
                                    # html.Span(id="html-a-sphere-output")
                                ], width={'size': 6, 'offset': 0}),
                                dbc.Col([
                                    html.Div([
                                        html.Img(src=app.get_asset_url('obj_bunny.png'), id='html-a-bunny',
                                                 height="80%", style={'margin': 'auto'}),
                                    ], id='html-bunny-block-color', className='box_objects'),
                                    # html.Span(id="html-a-bunny-output"),
                                ], width={'size': 6, 'offset': 0}),
                            ], align='center'),
                        ]),
                        className='box',
                        style={'width': '100%', 'margin-bottom': '20px',
                               'padding-top': '15px', 'padding-bottom': '15px'}
                    ),
                    html.Div([
                        dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    daq.PowerButton(on='True', id='powerbt-level-first', size=60, color='#4b9072')
                                ], width={'size': 4, 'offset': 0}, style={'padding-left': '15px'}),
                                dbc.Col([
                                    daq.PowerButton(id='powerbt-level-second', size=60, color='#4b9072')
                                ], width={'size': 4, 'offset': 0}, id='', style={'padding-left': '15px'}),
                                dbc.Col([
                                    daq.PowerButton(id='powerbt-level-third', size=60, color='#4b9072')
                                ], width={'size': 4, 'offset': 0}, style={'padding-left': '15px'}),
                            ], align='center'),
                        ]),
                        dbc.Container([
                            dbc.Button([dbc.Spinner(size="sm", children=[html.Div(id="loading-estimate")])],
                                       color="success", outline=False, id='bt-start-estimate',
                                       style={'background-color': '#4b9072', 'color': 'white'}, block=True)
                        ], fluid=True, style={'margin-top': '20px'}),
                    ], className='box',
                        style={'width': '100%', 'margin-bottom': '20px',
                               'padding-top': '25px', 'padding-bottom': '25px'}
                    ),

                    html.Div([
                        dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    daq.LEDDisplay(id='axios-x-output', value='0000', size=25, color='#4b9072'),
                                ], width={'size': 6, 'offset': 0}, style={'padding-left': '15px'}),
                                dbc.Col([
                                    daq.LEDDisplay(id='axios-y-output', value='0000', size=25, color='#4b9072'),
                                ], width={'size': 6, 'offset': 0}, style={'padding-left': '15px'}),
                            ], align='center'),
                        ]),
                        dbc.Container([
                            dbc.ButtonGroup([
                                dbc.Button([dbc.Spinner(size="sm", children=[html.Div(id="loading-render")])],
                                           color="success", id='bt-start-render',
                                           style={'background-color': '#4b9072', 'color': 'white'}),
                                # dbc.Button([dbc.Spinner(size="sm", children=[html.Div(id="loading-show")])],
                                #            color="success", id='bt-start-show',
                                #            style={'background-color': '#4b9072', 'color': 'white'})

                            ], style={'width': '100%'}),
                        ], fluid=True, style={'margin-top': '20px'}),
                    ], className='box',
                        style={'width': '100%', 'padding-top': '25px', 'padding-bottom': '25px'}
                    ),

                ]
            ),

        ]),
    ], fluid=True, style={'width': '95%', 'margin-top': '5px'}),
    html.Div(id="test-output"),
    html.Div([
        make_tooltip('Size of Object', 'knob-obj-size-output'),
        make_tooltip('Roughness of Object', 'knob-obj-roughness-output'),
        make_tooltip('R', 'color-R-output', 'left'),
        make_tooltip('G', 'color-G-output', 'left'),
        make_tooltip('B', 'color-B-output', 'left'),
        make_tooltip('Sphere', 'html-a-sphere'),
        make_tooltip('Bunny', 'html-a-bunny'),
        make_tooltip('Low', 'powerbt-level-first    '),
        make_tooltip('Normal', 'powerbt-level-second'),
        make_tooltip('BS Layer', 'powerbt-level-third'),
        make_tooltip('X of Position', 'axios-x-output'),
        make_tooltip('Y of Position', 'axios-y-output'),
        make_tooltip('Viewer of Video', 'view-tabs'),
        make_tooltip('Viewer of Image', 'image-tabs'),
        make_tooltip('Viewer of Chart', 'sunburst-tabs'),
        make_tooltip('Mode of Rendering', 'radio-container', 'bottom')
        # make_tooltip('Choose the Mode of Rendering: &#013;'
        #              'Memory On/Off: Save or not save results of rendering'
        #              'Multi Obj: Support to render a series of objects between the two points you picked.',
        #              'radio-container', 'bottom'),
    ])
])
