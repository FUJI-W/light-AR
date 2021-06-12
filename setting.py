import os
import os.path as osp
import pathlib

FOV_X = 63.4149  # the field of view in _x axis
COLOR_R, COLOR_G, COLOR_B = 0.8, 0.8, 0.8  # the diffuse color of the inserted object
COLOR = [COLOR_R, COLOR_G, COLOR_B]
ROUGHNESS = 0.8  # the roughness value of the inserted object.
SCALE = 0.2  # the size of the inserted object
ROOT = ''
PHOTO = 'im.png'
ESTIMATE_MODE = 1  # 0; 1; 2

RENDER = './render/build-5R/bin/optixRenderer'

PATH_APP = str(pathlib.Path(__file__).parent.resolve())

PATH_IN = osp.join(PATH_APP, 'inout/inputs', ROOT)
PATH_OUT = osp.join(PATH_APP, 'inout/outputs', ROOT)

PATH_ENV_MAP = osp.join(PATH_IN, 'envmap.png.npz')
PATH_DIFFUSE = osp.join(PATH_IN, 'albedo.png')
PATH_ROUGH = osp.join(PATH_IN, 'rough.png')
PATH_NORMAL = osp.join(PATH_IN, 'normal.npy')

PATH_LOG = osp.join(PATH_OUT, 'out.log')
PATH_LOG_EST = osp.join(PATH_OUT, 'out.est.log')
PATH_LOG_GEN = osp.join(PATH_OUT, 'out.gen.log')

