import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_uploader as du
import dash_daq as daq

from dash.dependencies import Input, Output, State
from server import app


from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import plotly.express as px
import pathlib
import os.path as osp
import numpy as np
import cv2

from setting import *


def make_data_tooltip(_label, _target, _placement='top'):
    return dbc.Tooltip(
        _label, target=_target, placement=_placement,
        hide_arrow=True,
        style={"background-color": 'rgba(75, 144, 144, 0.2)', "color": 'rgba(75, 144, 144, 1)'}
    )


data_graph_config = {'displayModeBar': True, 'scrollZoom': False, 'displaylogo': False}


def get_data_graph_figure(_img, _is_gray=False, _height=300, _margin=None):
    if _margin is None:
        _margin = dict(l=0, r=0, b=0, t=30)
    if _is_gray:
        fig = px.imshow(_img, color_continuous_scale='gray')
    else:
        fig = px.imshow(_img)

    fig.update_layout(
        # autosize=True,
        height=_height,
        margin=_margin,
        paper_bgcolor='#f9f9f8',
        plot_bgcolor='#f9f9f8',
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    if _is_gray:
        fig.update_layout(
            coloraxis_showscale=False,
        )
    return fig


def get_envmap_figure(_path):
    def writeEnvToFile(envmap, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
        envRow, envCol = envmap.shape[0], envmap.shape[1]

        interY = int(envRow / nrows)
        interX = int(envCol / ncols)

        lnrows = len(np.arange(0, envRow, interY))
        lncols = len(np.arange(0, envCol, interX))

        lenvHeight = lnrows * (envHeight + gap) + gap
        lenvWidth = lncols * (envWidth + gap) + gap

        envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
        for r in range(0, envRow, interY):
            for c in range(0, envCol, interX):
                rId = int(r / interY)
                cId = int(c / interX)

                rs = rId * (envHeight + gap)
                cs = cId * (envWidth + gap)
                envmapLarge[rs: rs + envHeight, cs: cs + envWidth, :] = envmap[r, c, :, :, :]

        envmapLarge = np.clip(envmapLarge, 0, 1)
        envmapLarge = (255 * (envmapLarge ** (1.0 / 2.2))).astype(np.uint8)
        cv2.imwrite(envName, envmapLarge[:, :, ::-1])
        return envmapLarge[:, :, ::-1]

    npz = np.load(_path)['env']
    _img = writeEnvToFile(envmap=npz, envName=osp.join(PATH_IN, 'envmap.png'),
                          nrows=npz.shape[0], ncols=npz.shape[1],
                          envHeight=npz.shape[2], envWidth=npz.shape[3])
    return get_data_graph_figure(_img, _height=350)


def get_rgbe_figure(_path):
    hdr = cv2.imread(_path, cv2.IMREAD_ANYDEPTH)
    hdr = np.maximum(hdr, 0)
    ldr = hdr ** (1.0 / 2.2)
    ldr = np.minimum(ldr * 255, 255)
    ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
    return get_data_graph_figure(ldr, _height=245)


@app.callback(
    [
        Output('graph-input', 'figure'),
        Output('graph-light', 'figure'),
        Output('graph-normal', 'figure'),
        Output('graph-rough', 'figure'),
        Output('graph-albedo', 'figure'),
        Output('graph-output', 'figure'),
        Output('graph-scene-rgbe', 'figure'),
        Output('graph-obj-mask', 'figure'),
        Output('graph-bkg-rgbe', 'figure'),
        Output('graph-scene-mask', 'figure'),
    ],
    [
        Input('navlink-data', 'n_clicks')
    ],
    [
        State('graph-input', 'figure'),
        State('graph-light', 'figure'),
        State('graph-normal', 'figure'),
        State('graph-rough', 'figure'),
        State('graph-albedo', 'figure'),
        State('graph-output', 'figure'),
        State('graph-scene-rgbe', 'figure'),
        State('graph-obj-mask', 'figure'),
        State('graph-bkg-rgbe', 'figure'),
        State('graph-scene-mask', 'figure'),
    ],
)
def update_data_page(
        n, input, light, normal, rough, albedo,
        output, scene_rgbe, obj_mask, bkg_rgbe, scene_mask
):
    if n is not None:
        input = get_data_graph_figure(Image.open(osp.join(PATH_IN, 'im.png')))
        light = get_envmap_figure(osp.join(PATH_IN, 'envmap.png.npz'))
        normal = get_data_graph_figure(Image.open(osp.join(PATH_IN, 'normal.png')))
        rough = get_data_graph_figure(Image.open(osp.join(PATH_IN, 'rough.png')), True)
        albedo = get_data_graph_figure(Image.open(osp.join(PATH_IN, 'albedo.png')))

        output = get_data_graph_figure(Image.open(osp.join(PATH_OUT, 'im.png')), _height=500)
        scene_rgbe = get_rgbe_figure(osp.join(PATH_OUT, 'scene_1.rgbe'))
        obj_mask = get_data_graph_figure(Image.open(osp.join(PATH_OUT, 'scene_objmask_1.png')), _height=245)
        bkg_rgbe = get_rgbe_figure(osp.join(PATH_OUT, 'scene_bkg_1.rgbe'))
        scene_mask = get_data_graph_figure(Image.open(osp.join(PATH_OUT, 'scenemask_1.png')), _height=245)
    return input, light, normal, rough, albedo, output, scene_rgbe, obj_mask, bkg_rgbe, scene_mask
