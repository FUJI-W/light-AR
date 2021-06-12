import dash_core_components as dcc
import dash_uploader as du
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from server import app, server

import flask
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import os.path as osp

from paint import *

du.configure_upload(app, folder='data')


def get_color_picker_figure():
    color_fig = px.imshow(Image.open(os.path.join(PATH_APP, "assets", 'color-picker.png')))
    color_fig.update_layout(
        # autosize=True,
        height=200,
        margin=dict(l=5, r=10, b=0, t=0, pad=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis={
            # 'range': [0.2, 1],
            'showgrid': False,  # thin lines in the background
            'zeroline': False,  # thick line at x=0
            'visible': False,  # numbers below
        },
        yaxis={
            # 'range': [0.2, 1],
            'showgrid': False,  # thin lines in the background
            'zeroline': False,  # thick line at x=0
            'visible': False,  # numbers below
        },
    )
    return color_fig


def get_graph_figure(_img):
    fig = px.imshow(_img)
    fig.update_layout(
        autosize=True,
        # height=800,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        paper_bgcolor='#f9f9f8',
        plot_bgcolor='#f9f9f8',
        showlegend=False,
        xaxis={
            # 'range': [0.2, 1],
            'showgrid': False,  # thin lines in the background
            'zeroline': False,  # thick line at x=0
            'visible': False,  # numbers below
        },
        yaxis={
            # 'range': [0.2, 1],
            'showgrid': False,  # thin lines in the background
            'zeroline': False,  # thick line at x=0
            'visible': False,  # numbers below
        },
    )
    return fig


def get_dict_from_log(logfile='out.log'):
    with open(logfile, 'r') as f:
        lines = [[i for i in line.split()] for line in f.readlines()]
    words = dict()
    for _l in lines:
        if _l[0] not in words.keys():
            words[_l[0].strip(':')] = []
        words[_l[0].strip(':')].append(float(_l[1]))
    means = dict()
    for _w in words:
        means[_w] = np.mean(np.array(words.get(_w)))
    return means


def get_chart_figure(logfile='out.log'):
    means = get_dict_from_log(logfile)
    y = list(means.values())
    x = list(means.keys())

    trace1 = go.Bar(
        x=x,
        y=y,
        name="Total Process"
    )

    layout = go.Layout(
        title='Total Process',
    )

    return go.Figure(
        data=[trace1],
        layout=layout
    )


def get_sunburst_figure():
    dict_est = get_dict_from_log(osp.join(PATH_OUT, 'out.est.log'))
    dict_gen = get_dict_from_log(osp.join(PATH_OUT, 'out.gen.log'))
    dict_all = get_dict_from_log(osp.join(PATH_OUT, 'out.log'))

    methods = list()
    modules = list()
    data = list()

    for k in dict_est:
        methods.append(k)
        modules.append('estimate_scene')
        data.append(dict_est[k])

    for k in dict_gen:
        methods.append(k)
        modules.append('info_process')
        data.append(dict_gen[k])

    for k in dict_all:
        if k == 'estimate_scene' or k == 'render_img' or k == 'generate_render_xml':
            continue
        methods.append(k)
        modules.append('info_process')
        data.append(dict_all[k])

    methods.append('render_img')
    modules.append('render_img')
    data.append(dict_all['render_img'])

    df = pd.DataFrame(dict(methods=methods, modules=modules, cost_time=data))

    fig = px.sunburst(df, path=['modules', 'methods'], values='cost_time', color='methods',
                      color_discrete_sequence=px.colors.sequential.haline_r)
    fig.update_traces(hovertemplate='%{label}<br>' + 'cost time: %{value} ms', textinfo="label + percent entry")

    fig.update_layout(
        autosize=True,
        # height=400,
        margin=dict(l=0, r=0, b=30, t=30, pad=0),
        paper_bgcolor='#f9f9f8',
        plot_bgcolor='#f9f9f8',
        # showlegend=False,
    )

    return fig


def make_tooltip(_label, _target, _placement='top'):
    return dbc.Tooltip(
        _label, target=_target, placement=_placement,
        hide_arrow=True,
        style={"background-color": 'rgba(75, 144, 144, 0.2)', "color": 'rgba(75, 144, 144, 1)'}
    )


@app.callback(
    Output('knob-obj-size-output', 'value'),
    [Input('knob-obj-size', 'value')]
)
def update_knob_obj_size_output(value):
    value = int(value)
    if value >= 100:
        value = 99
    return str(value).zfill(2)


@app.callback(
    Output('knob-obj-roughness-output', 'value'),
    [Input('knob-obj-roughness', 'value')]
)
def update_knob_obj_size_output(value):
    value = int(value)
    if value >= 100:
        value = 99
    return str(value).zfill(2)


@app.callback(
    [
        Output("color-R-output", "value"),
        Output("color-G-output", "value"),
        Output("color-B-output", "value")
    ],
    [Input("color-picker", "clickData")],
    prevent_initial_call=True,
)
def update_color_output(click_data):
    if click_data:
        return str(click_data['points'][0]['color']['0']).zfill(3), \
               str(click_data['points'][0]['color']['1']).zfill(3), \
               str(click_data['points'][0]['color']['2']).zfill(3)
    else:
        return '255', '255', '255'


@app.callback(
    Output("html-sphere-block-color", "style"),
    [Input("html-a-sphere", "n_clicks")]
)
def on_sphere_click(n):
    style = dict()
    if n is None or (n % 2) == 0:
        return style
    else:
        style['box-shadow'] = '2px 2px 2px lightgrey'
        style['background-color'] = '#4b9072'
        return style


@app.callback(
    Output("html-bunny-block-color", "style"),
    [Input("html-a-bunny", "n_clicks"),
     Input("html-a-sphere", "n_clicks")]
)
def on_bunny_click(n_bunny, n_sphere):
    style = dict()
    if n_bunny is None and n_sphere is None:
        style['box-shadow'] = '2px 2px 2px lightgrey'
        style['background-color'] = '#4b9072'
        return style
    elif n_bunny is None or (n_bunny % 2) == 0:
        return style
    else:
        style['box-shadow'] = '2px 2px 2px lightgrey'
        style['background-color'] = '#4b9072'
        return style


@app.callback(
    [Output("axios-x-output", "value"), Output("axios-y-output", "value")],
    [Input("graph-picture", "clickData")],
    prevent_initial_call=True,
)
def update_axios_output(click_data):
    if click_data:
        return str(click_data['points'][0]['x']).zfill(4), str(click_data['points'][0]['y']).zfill(4)
    else:
        return '0000', '0000'


@du.callback(
    Output("div-graph-picture", "children"),
    id='uploader',
)
def update_graph_picture(filenames):
    img = Image.open(os.path.join(PATH_APP, filenames[0]))
    fig = get_graph_figure(img)
    return [
        dcc.Graph(
            id='graph-picture',
            figure=fig,
            config={
                # 'displayModeBar': True,
                'editable': False,
                'scrollZoom': False,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
                                           'autoScale2d', 'resetScale2d',
                                           'hoverClosestCartesian', 'hoverCompareCartesian',
                                           'zoomInGeo', 'zoomOutGeo', 'resetGeo', 'hoverClosestGeo',
                                           'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews',
                                           'toggleSpikelines', 'resetViewMapbox']
            }
        )
    ]


@app.callback(
    [Output("loading-estimate", "children")],
    [Input("bt-start-estimate", "n_clicks")],
    [
        State('uploader', 'fileNames'),
        State("powerbt-level-first", "on"),
        State("powerbt-level-second", "on"),
        State("powerbt-level-third", "on"),
    ]
)
def on_bt_start_estimate_click(
        n, img_name, bt_level1, bt_level2, bt_level3
):
    if n is None:
        return ["Start Estimate"]

    inputImg = Image.open(osp.join(PATH_APP, 'data/inputs', img_name[0]))
    inputImg.save(osp.join(PATH_IN, 'im.png'))
    imgH, imgW = inputImg.height, inputImg.width

    mode = 1 if bt_level2 else 0
    mode = 2 if bt_level3 else mode
    estimate_scene(
        _root=ROOT, _photo=PHOTO, _h=imgH, _w=imgW, _mode=mode
    )
    return ["Estimate Done"]


N = 0
CLICK_TIME = 0
X_OLD = 0
Y_OLD = 0


@app.callback(
    [
        Output("loading-render", "children"),
        Output("test-output", 'children'),
        Output("graph-picture", "figure")
    ],
    [
        Input("bt-start-render", 'n_clicks')
    ],
    [
        State('knob-obj-size', 'value'),
        State('knob-obj-roughness', 'value'),
        State("color-picker", "clickData"),
        State('uploader', 'fileNames'),
        State("html-a-sphere", "n_clicks"),
        State("html-a-bunny", "n_clicks"),
        State("powerbt-level-first", "on"),
        State("powerbt-level-second", "on"),
        State("powerbt-level-third", "on"),
        State("graph-picture", "clickData"),
        State("graph-picture", "figure"),
        State("radios-system-mode", "value")
    ]
)
def on_bt_start_render_click(
        n, obj_size, obj_roughness, color_picker,
        img_name, obj_sphere, obj_bunny, bt_level1, bt_level2, bt_level3, img_click, img_fig, radio_value
):
    global N, CLICK_TIME, X_OLD, Y_OLD

    if n is None or n == N:
        # return "Render", "Not Clicked", img_fig
        return "Start Render", " ", img_fig

    N = n

    inputImg = Image.open(osp.join(PATH_IN, 'im.png'))
    imgH, imgW = inputImg.height, inputImg.width

    out = dict()
    out['obj_size'] = obj_size / 100 if obj_size else SCALE
    out['obj_roughness'] = obj_roughness / 100 if obj_roughness else ROUGHNESS
    out['color_picker'] = [color_picker['points'][0]['color']['0'], color_picker['points'][0]['color']['1'],
                           color_picker['points'][0]['color']['2']] if color_picker else COLOR
    out['img_name'] = img_name[0]
    out['obj_sphere'] = obj_sphere % 2 if obj_sphere else 0
    out['obj_bunny'] = obj_bunny % 2 if obj_bunny else 1
    out['obj_name'] = 'sphere.obj' if out['obj_sphere'] else 'bunny.ply'
    out['bt_level'] = {'l0': bt_level1, 'l1': bt_level2, 'l2': bt_level3}
    out['img_click'] = {'x': img_click['points'][0]['x'],
                        'y': img_click['points'][0]['y']} if img_click else {'x': int(imgW / 2), 'y': int(imgH / 2)}

    if radio_value == 3:
        if CLICK_TIME % 2 == 0:
            CLICK_TIME += 1
            X_OLD, Y_OLD = out['img_click']['x'], out['img_click']['y']
            # return "Push Again", f"{CLICK_TIME}, {X_OLD},{Y_OLD}", img_fig
            return "Push Again", "", img_fig

        points = get_equal_dis_points(p1=[X_OLD, Y_OLD],
                                      p2=[out['img_click']['x'], out['img_click']['y']],
                                      step=1)

        for i in range(points.shape[0]):
            gui_object_insert(
                _img_h=imgH, _img_w=imgW,
                _obj_name=out['obj_name'],
                _obj_x=points[i][0],
                _obj_y=points[i][1],
                _offset=30 * imgW / 640 * out['obj_size'] / 0.2,
                _scale=out['obj_size'],
                _roughness=out['obj_roughness'],
                _diffuse=out['color_picker'],
                _index=i
            )

        convert_image_to_video(
            _num=points.shape[0],
            _root=PATH_OUT,
            _out_path=PATH_OUT,
        )

        # return "Render Done", f"{CLICK_TIME}, {X_OLD}, {Y_OLD}, {out['img_click']}", img_fig
        return "Render Done", "", img_fig

    else:
        gui_object_insert(
            _img_h=imgH, _img_w=imgW,
            _obj_name=out['obj_name'],
            _obj_x=out['img_click']['x'],
            _obj_y=out['img_click']['y'],
            _offset=30 * imgW / 640 * out['obj_size'] / 0.2,
            _scale=out['obj_size'],
            _roughness=out['obj_roughness'],
            _diffuse=out['color_picker']
        )

        if radio_value == 2:
            os.system('cp ' + osp.join(PATH_OUT, PHOTO) + ' ' + osp.join(PATH_IN, PHOTO))

        fig = get_graph_figure(Image.open(os.path.join(PATH_OUT, PHOTO)))

        # return "Render", str(out), fig
        return f"Render Done", " ", fig


# @app.callback(
#     [Output("loading-show", "children"),
#      Output("graph-picture", "figure")],
#     [Input("bt-start-show", "n_clicks")],
#     [State("graph-picture", "figure")]
# )
# def on_bt_start_show_click(n, fig):
#     if n is None:
#         return "Show", fig
#
#     fig = get_graph_figure(Image.open(os.path.join(PATH_OUT, PHOTO)))
#     return "Show", fig


@server.route('/inout/outputs/<path:path>')
def serve_static(path):
    return flask.send_from_directory(os.path.join(PATH_APP, 'inout/outputs/'), path)


@app.callback(
    Output('movie-player', 'src'),
    Input('view-tabs', 'value')
)
def update_videos(v):
    src = "inout/outputs/video.webm"
    return src


@app.callback(
    Output('graph-sunburst', 'figure'),
    Input('sunburst-tabs', 'value')
)
def update_sunburst_figure(v):
    return get_sunburst_figure()
