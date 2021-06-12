import math
import os

import numpy as np

os.environ['NUMEXPR_MAX_THREADS'] = '40'

import scipy.io
from PIL import Image
from setting import *
from generate import *
from tools import *


@timeit(PATH_LOG)
def estimate_scene(_root, _photo, _h, _w, _mode=ESTIMATE_MODE):
    os.system('sh estimate.sh '
              ' -i inout/inputs' + _root + '/' + _photo +
              ' -o inout/inputs' + _root +
              ' -m ' + str(_mode) +
              ' -h ' + str(_h) +
              ' -w ' + str(_w)
              )


def pick_plane_points(_img):
    # plt.ion()
    plt.imshow(_img)
    _x, _y = zip(*plt.ginput(4))
    plt.close()
    _x = np.array([_x]).T
    _y = np.array([_y]).T
    return _x, _y


@timeit(PATH_LOG)
def generate_mask(_x, _y, _h, _w):
    _grid_x, _grid_y = np.meshgrid(range(1, _w + 1), range(1, _h + 1))
    _mask = np.ones((_h, _w))
    for i in range(4):
        _seg = [_x[(i + 1) % 4] - _x[i], _y[(i + 1) % 4] - _y[i]]
        _mask = np.multiply(_mask, ((_grid_x - _x[i]) * _seg[1] - (_grid_y - _y[i]) * _seg[0]) > 0)
    return _mask


@timeit(PATH_LOG)
def generate_3d_points(_x, _y, _mask):
    _normal = np.load(PATH_NORMAL)
    _h, _w = _normal.shape[0], _normal.shape[1]
    _normal_x = _normal[:, :, 0]
    _normal_y = _normal[:, :, 1]
    _normal_z = _normal[:, :, 2]
    # TODO: whether erode or not
    _mask_eroded = _mask
    # _mask_eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    _normal_x = np.mean(_normal_x[_mask_eroded == 1])
    _normal_y = np.mean(_normal_y[_mask_eroded == 1])
    _normal_z = np.mean(_normal_z[_mask_eroded == 1])
    _vn = np.array([[_normal_x], [_normal_y], [_normal_z]])
    _vn = _vn / np.sum(np.multiply(_vn, _vn)) ** 0.5
    _vn = np.multiply(_vn, [[-1], [1], [-1]])

    _atan_fov_x = math.tan(math.radians(FOV_X / 2.0))
    _atan_fov_y = _atan_fov_x / _w * _h
    _v = np.c_[(((_w + 1) / 2.0) - _x) / ((_w - 1) / 2.0) * _atan_fov_x,
               (((_h + 1) / 2.0) - _y) / ((_h - 1) / 2.0) * _atan_fov_y,
               np.ones((4, 1))]
    for i in range(1, 4):
        d = (_v[0] @ _vn) / (_v[i] @ _vn)
        assert d > 0
        _v[i] = d * _v[i]

    _vt = np.c_[(_x - 1) / (_w - 1), (_h - _y) / (_h - 1)]

    return _vn, _v, _vt


@timeit(PATH_LOG)
def output_3d_mesh(_vn, _v, _vt):
    _path_mesh = os.path.join(PATH_OUT, 'mesh.obj')  # path to plane mesh
    with open(_path_mesh, 'w') as f:
        for i in range(0, 4):
            f.write('v %.5f %.5f %.5f\n' % (_v[i][0], _v[i][1], _v[i][2]))
        for i in range(0, 4):
            f.write('vt %.5f %.5f\n' % (_vt[i][0], _vt[i][1]))
        f.write('vn %.5f %.5f %.5f\n' % (_vn[0], _vn[1], _vn[2]))
        f.write('f 1/1/1 2/2/1 3/3/1\n')
        f.write('f 1/1/1 3/3/1 4/4/1\n')
        f.close()


def pick_obj_point(_x, _y, _h, _w):
    _tx, _ty = np.r_[_x, [_x[0]]], np.r_[_y, [_y[0]]]
    plt.ion()
    _img = np.asarray(Image.open(os.path.join(PATH_IN, PHOTO)).resize((_w, _h), Image.ANTIALIAS))
    plt.imshow(_img)
    plt.plot(_tx, _ty, color='r', linestyle='--')
    plt.show()
    (_obj_x, _obj_y) = plt.ginput(1)[0]
    plt.close()
    return _obj_x, _obj_y


def pick_obj_points(_x, _y, _h, _w, _count=1):
    _tx, _ty = np.r_[_x, [_x[0]]], np.r_[_y, [_y[0]]]
    plt.ion()
    _img = np.asarray(Image.open(os.path.join(PATH_IN, PHOTO)).resize((_w, _h), Image.ANTIALIAS))
    plt.imshow(_img)
    plt.plot(_tx, _ty, color='r', linestyle='--')
    plt.show()
    _obj_points = plt.ginput(_count)
    plt.close()
    return _obj_points


@timeit(PATH_LOG)
def generate_info(_obj_x, _obj_y, _vn, _v, _h, _w, _scale=SCALE):
    _vImg = [(_obj_x - 1) / (_w - 1), (_obj_y - 1) / (_h - 1)]

    _atan_fov_x = math.tan(math.radians(FOV_X / 2.0))
    _atan_fov_y = _atan_fov_x / _w * _h
    _vObj = np.array([(((_w + 1) / 2.0) - _obj_x) / ((_w - 1) / 2.0) * _atan_fov_x,
                      (((_h + 1) / 2.0) - _obj_y) / ((_h - 1) / 2.0) * _atan_fov_y,
                      1])
    _d = (_v[0] @ _vn) / (_vObj @ _vn)
    assert _d > 0
    _vObj = _vObj * _d

    # TODO: check if it is different from the .m
    # _dist = np.c_[np.linalg.norm(_vObj - _v[0]), np.linalg.norm(_vObj - _v[1])]
    # _scale = SCALE * np.min(_dist)
    _scale = _scale * 0.1

    _path_info = os.path.join(PATH_OUT, 'info.mat')  # the starting point
    scipy.io.savemat(_path_info, {'vn': _vn, 'vObj': _vObj, 'scale': _scale, 'vImg': _vImg})


@timeit(PATH_LOG)
def generate_render_xml(_path_obj, _h, _w, _roughness=ROUGHNESS, _diffuse=None):
    if _diffuse is None:
        _diffuse = [COLOR_R, COLOR_G, COLOR_B]

    _path_env = PATH_ENV_MAP  # path to environmental map
    _path_env_mat_origin = os.path.join(PATH_OUT, 'envOrigin.mat')
    _path_env_mat = os.path.join(PATH_OUT, 'env.mat')

    # Load information
    _info = scipy.io.loadmat(os.path.join(PATH_OUT, 'info.mat'))
    _vn, _vObj, _vImg, _scale = _info['vn'], _info['vObj'], _info['vImg'], _info['scale']
    _vn, _vObj, _vImg = _vn.flatten(), _vObj.flatten(), _vImg.flatten()
    _vn = _vn / np.sqrt(np.sum(_vn * _vn))

    # Load environmental map
    _env = np.load(_path_env)['env']
    _env_row, _env_col = _env.shape[0], _env.shape[1]
    _rId, _cId = (_env_row - 1) * _vImg[1], (_env_col - 1) * _vImg[0]
    _rId = np.clip(np.round(_rId), 0, _env_row - 1)
    _cId = np.clip(np.round(_cId), 0, _env_col - 1)
    _env = _env[int(_rId), int(_cId), :, :, :]
    # _env = cv2.resize(_env, (2048, 512), interpolation=cv2.INTER_LINEAR)
    _env = cv2.resize(_env, (1024, 256), interpolation=cv2.INTER_LINEAR)
    scipy.io.savemat(_path_env_mat_origin, {'env': np.maximum(_env, 0)})
    # TODO: where the time costs
    # _env_balck = np.zeros([512, 2048, 3], dtype=np.float32)
    _env_balck = np.zeros([256, 1024, 3], dtype=np.float32)
    _env = np.concatenate([_env, _env_balck], axis=0)
    # _env = rotateEnvmap_bkq(_env, _vn)
    # _env = rotateEnvmap_OPT_NUMBA(_env, _vn)
    # _env = rotateEnvmap_OPT_MATRIX(_env, _vn)
    # _env = rotateEnvmap_OPT_MATRIX_NE(_env, _vn)
    _env = rotateEnvmap_OPT_CUDA(_env, _vn)
    scipy.io.savemat(_path_env_mat, {'env': np.maximum(_env, 0)})

    # Build the materials for the two shapes
    _shapes = [os.path.join(PATH_OUT, 'mesh.obj'),
               os.path.join('models', _path_obj)]
    _mat1 = mat(texture=tex(diffuseName=PATH_DIFFUSE,
                            roughnessName=PATH_ROUGH))
    _mat2 = mat(diffuse=_diffuse, roughness=_roughness)
    _materials = [_mat1, _mat2]

    _path_obj = os.path.join('models', _path_obj)  # path to the new object mesh
    _mesh_info = scipy.io.loadmat(os.path.splitext(_path_obj)[0] + 'Init.mat')
    _mesh_rotateAxis = _mesh_info.get('meshRotateAxis')[0]
    _mesh_rotateAxis = np.array(_mesh_rotateAxis, dtype=np.float32)
    _mesh_rotateAxis = _mesh_rotateAxis / np.sqrt(np.sum(_mesh_rotateAxis * _mesh_rotateAxis))

    _up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _rotate_axis = np.cross(_up, _vn)
    if np.sum(_rotate_axis * _rotate_axis) <= 1e-6:
        _rotate_axis = None
        _rotate_angle = None
    else:
        _rotate_axis = _rotate_axis / np.sqrt(np.sum(_rotate_axis * _rotate_axis))
        _rotate_angle = np.arccos(np.sum(_vn * _up))

    generateXML(
        shapes=_shapes, materials=_materials,

        envmapName=os.path.join(PATH_OUT, 'env.hdr'), xmlName='scene.xml',
        sampleCount=1024, imWidth=_w, imHeight=_h, fovValue=FOV_X,

        meshRotateAxis=_mesh_rotateAxis,
        meshRotateAngle=_mesh_info.get('meshRotateAngle')[0],
        meshTranslate=_mesh_info.get('meshTranslate')[0],
        meshScale=_mesh_info.get('meshScale')[0],

        rotateAxis=_rotate_axis,
        rotateAngle=_rotate_angle,
        translation=_vObj,
        scale=_scale
    )


@timeit(PATH_LOG)
def render_img():
    _env = scipy.io.loadmat(os.path.join(PATH_OUT, 'env.mat')).get('env')
    # TODO: Whether flip or not
    # _env = cv2.flip(_env, 0, dst=None)
    cv2.imwrite(os.path.join(PATH_OUT, 'env.hdr'), np.maximum(_env, 0))

    # TODO: Change the structure of files
    _xml_file1, _output1 = 'scene.xml', 'scene.rgbe'
    _xml_file2, _output2 = 'scene_obj.xml', 'scene_obj.rgbe'
    _xml_file3, _output3 = 'scene_bkg.xml', 'scene_bkg.rgbe'

    os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (_xml_file1, _output1, 0))
    # os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (_xml_file2, _output2, 0))
    os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (_xml_file3, _output3, 0))
    os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (_xml_file1, _output1, 4))
    os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (_xml_file2, _output2, 4))
    #os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (PATH_XML1, PATH_RENDER_OUT1, 0))
    #os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (PATH_XML3, PATH_RENDER_OUT3, 0))
    #os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (PATH_XML1, PATH_RENDER_OUT1, 4))
    #os.system(RENDER + ' -f %s -o %s -m %d --gpuIds 2' % (PATH_XML2, PATH_RENDER_OUT2, 4))


@timeit(PATH_LOG)
def mix_up_img(_h, _w):
    #_hdr1 = cv2.imread(PATH_SCENE_RGBE, cv2.IMREAD_ANYDEPTH)
    #_hdr2 = cv2.imread(PATH_BJG_RGBE, cv2.IMREAD_ANYDEPTH)
    #_mask1 = np.asarray(Image.open(PATH_SCENE_MASK)) / 255.0
    #_mask2 = np.asarray(Image.open(PATH_OBJ_MASK)) / 255.0
    
    _hdr1 = cv2.imread('scene_1.rgbe', cv2.IMREAD_ANYDEPTH)
    _hdr2 = cv2.imread('scene_bkg_1.rgbe', cv2.IMREAD_ANYDEPTH)
    _mask1 = np.asarray(Image.open('scenemask_1.png')) / 255.0
    _mask2 = np.asarray(Image.open('scene_objmask_1.png')) / 255.0

    _maskBg = np.maximum(_mask1 - _mask2, 0)
    
    _diff = np.maximum(_hdr1, 1e-10) / np.maximum(_hdr2, 1e-10) * _maskBg
    _diff = np.minimum(_diff * 1.01, 1)

    _ldr = cv2.resize(cv2.imread(os.path.join(PATH_IN, PHOTO)) / 255.0, (_w, _h))
    _hdr = _ldr ** 2.2

    # TODO: whether erode or not
    _mask1 = cv2.erode(_mask1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    _hdr_new = _hdr * (1 - _mask1) + (_hdr * _diff) * _mask1
    _hdr_new = _hdr1 * _mask2 + _hdr_new * (1 - _mask2)
    _hdr_new = np.maximum(_hdr_new, 0)
    _ldr_new = _hdr_new ** (1.0 / 2.2)

    return _hdr_new, _ldr_new


@timeit(PATH_LOG)
def mix_up_img_simple(_h, _w, _mask):
    _hdr_scene = cv2.imread('scene_1.rgbe', cv2.IMREAD_ANYDEPTH)
    _ldr = cv2.resize(cv2.imread(os.path.join(PATH_IN, PHOTO)) / 255.0, (_w, _h))
    _hdr = _ldr ** 2.2

    _mask = np.repeat(_mask[:, :, np.newaxis], 3, axis=2)
    _hdr_new = _hdr * (1 - _mask) + _hdr_scene * _mask
    _hdr_new = np.maximum(_hdr_new, 0)
    _ldr_new = _hdr_new ** (1.0 / 2.2)

    cv2.imshow("", _ldr_new.astype(np.float64))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return _hdr_new, _ldr_new


def get_equal_dis_points(p1, p2, step=1):
    parts = int(max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])) / step)
    points = list(zip(np.linspace(p1[0], p2[0], parts + 1), np.linspace(p1[1], p2[1], parts + 1)))
    return np.asarray(points).astype(int)


def gui_object_insert(_img_h, _img_w, _obj_name, _obj_x, _obj_y, _offset, _scale, _roughness, _diffuse, _index=-1):
    _x = np.array([[_obj_x - 3 * _offset],
                   [_obj_x - _offset],
                   [_obj_x + 3 * _offset],
                   [_obj_x + _offset]])
    _y = np.array([[_obj_y - _offset],
                   [_obj_y + 2 * _offset],
                   [_obj_y + _offset],
                   [_obj_y - 2 * _offset]])
    _mask = generate_mask(_x, _y, _img_h, _img_w)
    _vn, _v, _vt = generate_3d_points(_x, _y, _mask)
    output_3d_mesh(_vn, _v, _vt)
    generate_info(_obj_x, _obj_y, _vn, _v, _img_h, _img_w, _scale=_scale)
    generate_render_xml(_obj_name, _img_h, _img_w, _roughness=_roughness, _diffuse=_diffuse)
    render_img()
    _hdr_New, _ldr_New = mix_up_img(_img_h, _img_w)
    if _index == -1:
        cv2.imwrite(os.path.join(PATH_OUT, 'im.png'), np.minimum(_ldr_New * 255, 255))
    else:
        cv2.imwrite(os.path.join(PATH_OUT, f'im{_index}.png'), np.minimum(_ldr_New * 255, 255))

    os.system('mv *.xml *.png *.rgbe ' + PATH_OUT)

    return _ldr_New


def convert_image_to_video(_num, _root='', _name='im', _out_path='', _video_name='video', _fps=12):
    img = cv2.imread(osp.join(_root, f'{_name}0.png'))
    size = (img.shape[1], img.shape[0])
    # fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    video = cv2.VideoWriter(osp.join(_out_path, f'{_video_name}.webm'), fourcc, _fps, size)

    for i in range(1, _num):
        img = cv2.imread(osp.join(_root, f'{_name}{i}.png'))
        video.write(img)

    print('Convert to Video Finished!')
