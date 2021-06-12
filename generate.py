import os
import xml.etree.ElementTree as et
from xml.dom import minidom
import numpy as np
import argparse
import glob
import cv2
import math

import numba
from numba import cuda

import numexpr as ne

from setting import *
from tools import *



class tex():
    def __init__(self, diffuseName=None, roughnessName=None):
        self.diffuseName = diffuseName
        self.roughnessName = roughnessName


class mat():
    def __init__(self, name='mat', diffuse=None, roughness=None, texture=None):
        self.name = name
        self.diffuse = diffuse
        self.roughness = roughness
        self.texture = texture


############################# Code for Generating the Xml file ########################
@timeit(PATH_LOG_GEN)
def addShape(root, name, materials, isAddSpecular=True,
             isAddTransform=False,
             meshTranslate=None, meshRotateAxis=None, meshRotateAngle=None, meshScale=None,
             rotateAxis=None, rotateAngle=None,
             scaleValue=None, translationValue=None):
    shape = et.SubElement(root, 'shape')
    shape.set('id', '{0}_object'.format(name.split('.')[0]))

    objType = name.split('.')[-1]
    assert (objType == 'ply' or objType == 'obj')
    shape.set('type', objType)
    stringF = et.SubElement(shape, 'string')
    stringF.set('name', 'filename')
    stringF.set('value', name)

    for material in materials:
        bsdf = et.SubElement(shape, 'bsdf')

        if isAddSpecular == False:
            bsdf.set('type', 'diffuse')
            if material.texture is None:
                rgb = et.SubElement(bsdf, 'rgb')
                rgb.set('name', 'reflectance')
                rgb.set('value', '%.5f %.5f %.5f'
                        % (material.diffuse[0], material.diffuse[1], material.diffuse[2]))
            else:
                diffPath = material.texture.diffuseName
                texture = et.SubElement(bsdf, 'texture')
                texture.set('name', 'reflectance')
                texture.set('type', 'bitmap')
                filename = et.SubElement(texture, 'string')
                filename.set('name', 'filename')
                filename.set('value', diffPath)

        elif isAddSpecular == True:
            bsdf.set('type', 'microfacet')
            if material.texture is None:
                rgb = et.SubElement(bsdf, 'rgb')
                rgb.set('name', 'albedo')
                rgb.set('value', '%.5f %.5f %.5f'
                        % (material.diffuse[0], material.diffuse[1], material.diffuse[2]))
                rgb = et.SubElement(bsdf, 'float')
                rgb.set('name', 'roughness')
                rgb.set('value', '%.5f' % (material.roughness))
            else:
                diffPath = material.texture.diffuseName
                texture = et.SubElement(bsdf, 'texture')
                texture.set('name', 'albedo')
                texture.set('type', 'bitmap')
                filename = et.SubElement(texture, 'string')
                filename.set('name', 'filename')
                filename.set('value', diffPath)

                roughPath = material.texture.roughnessName
                texture = et.SubElement(bsdf, 'texture')
                texture.set('name', 'roughness')
                texture.set('type', 'bitmap')
                filename = et.SubElement(texture, 'string')
                filename.set('name', 'filename')
                filename.set('value', roughPath)

    if isAddTransform:
        transform = et.SubElement(shape, 'transform')
        transform.set('name', 'toWorld')
        if not meshTranslate is None:
            translation = et.SubElement(transform, 'translate')
            translation.set('x', '%.5f' % meshTranslate[0])
            translation.set('y', '%.5f' % meshTranslate[1])
            translation.set('z', '%.5f' % meshTranslate[2])
        if not meshRotateAxis is None:
            assert (not meshRotateAngle is None)
            rotation = et.SubElement(transform, 'rotate')
            rotation.set('x', '%.5f' % meshRotateAxis[0])
            rotation.set('y', '%.5f' % meshRotateAxis[1])
            rotation.set('z', '%.5f' % meshRotateAxis[2])
            rotation.set('angle', '%.5f' % meshRotateAngle)
        if not meshScale is None:
            scale = et.SubElement(transform, 'scale')
            scale.set('value', '%.5f' % meshScale)
        if not rotateAxis is None:
            assert (not rotateAngle is None)
            rotation = et.SubElement(transform, 'rotate')
            rotation.set('x', '%.5f' % rotateAxis[0])
            rotation.set('y', '%.5f' % rotateAxis[1])
            rotation.set('z', '%.5f' % rotateAxis[2])
            rotation.set('angle', '%.5f' % rotateAngle)
        if not scaleValue is None:
            scale = et.SubElement(transform, 'scale')
            scale.set('value', '%.5f' % scaleValue)
        if not translationValue is None:
            translation = et.SubElement(transform, 'translate')
            translation.set('x', '%.5f' % translationValue[0])
            translation.set('y', '%.5f' % translationValue[1])
            translation.set('z', '%.5f' % translationValue[2])
    return root


@timeit(PATH_LOG_GEN)
def addEnv(root, envmapName, scaleFloat):
    emitter = et.SubElement(root, 'emitter')
    emitter.set('type', 'envmap')
    filename = et.SubElement(emitter, 'string')
    filename.set('name', 'filename')
    filename.set('value', envmapName)
    scale = et.SubElement(emitter, 'float')
    scale.set('name', 'scale')
    scale.set('value', '%.4f' % (scaleFloat))
    return root


@timeit(PATH_LOG_GEN)
def addSensor(root, fovValue, imWidth, imHeight, sampleCount):
    camera = et.SubElement(root, 'sensor')
    camera.set('type', 'perspective')
    fov = et.SubElement(camera, 'float')
    fov.set('name', 'fov')
    fov.set('value', '%.4f' % (fovValue))
    fovAxis = et.SubElement(camera, 'string')
    fovAxis.set('name', 'fovAxis')
    fovAxis.set('value', 'x')
    transform = et.SubElement(camera, 'transform')
    transform.set('name', 'toWorld')
    lookAt = et.SubElement(transform, 'lookAt')
    lookAt.set('origin', '0 0 0')
    lookAt.set('target', '0 0 1.0')
    lookAt.set('up', '0 1.0 0')
    film = et.SubElement(camera, 'film')
    film.set('type', 'hdrfilm')
    width = et.SubElement(film, 'integer')
    width.set('name', 'width')
    width.set('value', '%d' % (imWidth))
    height = et.SubElement(film, 'integer')
    height.set('name', 'height')
    height.set('value', '%d' % (imHeight))
    sampler = et.SubElement(camera, 'sampler')
    sampler.set('type', 'adaptive')
    sampleNum = et.SubElement(sampler, 'integer')
    sampleNum.set('name', 'sampleCount')
    sampleNum.set('value', '%d' % (sampleCount))
    return root


@timeit(PATH_LOG_GEN)
def transformToXml(root):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString = xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0]
    xmlString = '\n'.join(xmlString)
    return xmlString


@timeit(PATH_LOG_GEN)
def generateXML(shapes, materials, envmapName, xmlName, sampleCount=1024,
                imWidth=640, imHeight=480, fovValue=63.4149,
                meshTranslate=None, meshRotateAxis=None, meshRotateAngle=None,
                meshScale=None, rotateAxis=None, rotateAngle=None,
                translation=None, scale=None):
    # Build the scene
    root = et.Element('scene')
    root.set('version', '0.5.0')
    integrator = et.SubElement(root, 'integrator')
    integrator.set('type', 'path')

    rootObj = et.Element('scene')
    rootObj.set('version', '0.5.0')
    integrator = et.SubElement(rootObj, 'integrator')
    integrator.set('type', 'path')

    rootBkg = et.Element('scene')
    rootBkg.set('version', '0.5.0')
    integrator = et.SubElement(rootBkg, 'integrator')
    integrator.set('type', 'path')

    ## Create the obj files that is not emitter
    # Write 3D meshes
    root = addShape(root, shapes[0], [materials[0]], True,
                    isAddTransform=False)
    rootBkg = addShape(rootBkg, shapes[0], [materials[0]], True,
                       isAddTransform=False)

    root = addShape(root, shapes[1], [materials[1]], True,
                    isAddTransform=True,
                    meshTranslate=meshTranslate, meshRotateAxis=meshRotateAxis,
                    meshRotateAngle=meshRotateAngle, meshScale=meshScale,
                    rotateAxis=rotateAxis, rotateAngle=rotateAngle,
                    scaleValue=scale, translationValue=translation)
    rootObj = addShape(rootObj, shapes[1], [materials[1]], True,
                       isAddTransform=True,
                       meshTranslate=meshTranslate, meshRotateAxis=meshRotateAxis,
                       meshRotateAngle=meshRotateAngle, meshScale=meshScale,
                       rotateAxis=rotateAxis, rotateAngle=rotateAngle,
                       scaleValue=scale, translationValue=translation)

    # Add the environmental map lighting
    root = addEnv(root, envmapName, 1)
    rootObj = addEnv(rootObj, envmapName, 1)
    rootBkg = addEnv(rootBkg, envmapName, 1)

    # Add the camera
    root = addSensor(root, fovValue, imWidth, imHeight, sampleCount)
    rootObj = addSensor(rootObj, fovValue, imWidth, imHeight, sampleCount)
    rootBkg = addSensor(rootBkg, fovValue, imWidth, imHeight, sampleCount)

    xmlString = transformToXml(root)
    xmlStringObj = transformToXml(rootObj)
    xmlStringBkg = transformToXml(rootBkg)

    with open(xmlName, 'w') as xmlOut:
        xmlOut.write(xmlString)
    with open(xmlName.replace('.xml', '_obj.xml'), 'w') as xmlOut:
        xmlOut.write(xmlStringObj)
    with open(xmlName.replace('.xml', '_bkg.xml'), 'w') as xmlOut:
        xmlOut.write(xmlStringBkg)


# Code for Rotating the Envmap
def angleToUV(theta, phi):
    u = (phi + np.pi) / 2 / np.pi
    v = 1 - theta / np.pi
    return u, v


def uvToEnvmap(envmap, u, v):
    height, width = envmap.shape[0], envmap.shape[1]
    c, r = u * (width - 1), (1 - v) * (height - 1)
    cs, rs = int(c), int(r)
    ce = min(width - 1, cs + 1)
    re = min(height - 1, rs + 1)
    wc, wr = c - cs, r - rs
    color1 = (1 - wc) * envmap[rs, cs, :] + wc * envmap[rs, ce, :]
    color2 = (1 - wc) * envmap[re, cs, :] + wc * envmap[re, ce, :]
    color = (1 - wr) * color1 + wr * color2
    return color


@timeit(PATH_LOG_GEN)
def rotateEnvmap_bkq(envmap, vn):
    up = np.array([0, 1, 0], dtype=np.float32)
    z = vn
    z = z / np.sqrt(np.sum(z * z))
    x = np.cross(up, z)
    x = x / np.sqrt(np.sum(x * x))
    y = np.cross(z, x)
    y = y / np.sqrt(np.sum(y * y))

    # x = np.asarray([x[2], x[0], x[1]], dtype = np.float32 )
    # y = np.asarray([y[2], y[0], y[1]], dtype = np.float32 )
    # z = np.asarray([z[2], z[0], z[1]], dtype = np.float32 )
    x, y, z = x[np.newaxis, :], y[np.newaxis, :], z[np.newaxis, :]

    R = np.concatenate([x, y, z], axis=0)
    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    print(R)

    envmapRot = np.zeros(envmap.shape, dtype=np.float32)
    height, width = envmapRot.shape[0], envmapRot.shape[1]
    for r in range(0, height):
        for c in range(0, width):
            theta = r / float(height - 1) * np.pi
            phi = (c / float(width) * np.pi * 2 - np.pi)
            z = np.sin(theta) * np.cos(phi)
            x = np.sin(theta) * np.sin(phi)
            y = np.cos(theta)
            coord = x * rx + y * ry + z * rz
            nx, ny, nz = coord[0], coord[1], coord[2]
            thetaNew = np.arccos(nz)
            nx = nx / (np.sqrt(1 - nz * nz) + 1e-12)
            ny = ny / (np.sqrt(1 - nz * nz) + 1e-12)
            nx = np.clip(nx, -1, 1)
            ny = np.clip(ny, -1, 1)
            nz = np.clip(nz, -1, 1)
            phiNew = np.arccos(nx)
            if ny < 0:
                phiNew = - phiNew
            u, v = angleToUV(thetaNew, phiNew)
            color = uvToEnvmap(envmap, u, v)
            envmapRot[r, c, :] = color

    return envmapRot


@numba.jit(nopython=True, parallel=True)
def rotate_OPT_NUMBA(envmap, R):
    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    envmapRot = np.zeros(envmap.shape, dtype=np.float32)
    height, width = envmapRot.shape[0], envmapRot.shape[1]
    for r in range(0, height):
        for c in range(0, width):
            theta = r / float(height - 1) * np.pi
            phi = (c / float(width) * np.pi * 2 - np.pi)
            z = np.sin(theta) * np.cos(phi)
            x = np.sin(theta) * np.sin(phi)
            y = np.cos(theta)
            coord = x * rx + y * ry + z * rz
            nx, ny, nz = coord[0], coord[1], coord[2]
            thetaNew = np.arccos(nz)
            nx = nx / (np.sqrt(1 - nz * nz) + 1e-12)
            ny = ny / (np.sqrt(1 - nz * nz) + 1e-12)
            # nx = np.clip(nx, -1, 1)
            # ny = np.clip(ny, -1, 1)
            # nz = np.clip(nz, -1, 1)
            phiNew = np.arccos(nx)
            if ny < 0:
                phiNew = - phiNew

            u = (phiNew + np.pi) / 2 / np.pi
            v = 1 - thetaNew / np.pi

            _c, _r = u * (width - 1), (1 - v) * (height - 1)
            _cs, _rs = int(_c), int(_r)
            _ce = min(width - 1, _cs + 1)
            _re = min(height - 1, _rs + 1)
            _wc, _wr = _c - _cs, _r - _rs
            _color1 = (1 - _wc) * envmap[_rs, _cs, :] + _wc * envmap[_rs, _ce, :]
            _color2 = (1 - _wc) * envmap[_re, _cs, :] + _wc * envmap[_re, _ce, :]
            color = (1 - _wr) * _color1 + _wr * _color2

            envmapRot[r, c, :] = color

    return envmapRot


@timeit(PATH_LOG_GEN)
def rotateEnvmap_OPT_NUMBA(envmap, vn):
    up = np.array([0, 1, 0], dtype=np.float32)
    z = vn
    z = z / np.sqrt(np.sum(z * z))
    x = np.cross(up, z)
    x = x / np.sqrt(np.sum(x * x))
    y = np.cross(z, x)
    y = y / np.sqrt(np.sum(y * y))
    x, y, z = x[np.newaxis, :], y[np.newaxis, :], z[np.newaxis, :]
    R = np.concatenate([x, y, z], axis=0)
    print(R)

    return rotate_OPT_NUMBA(envmap, R)


def rotate_OPT_MATRIX(envmap, height, width, _rs, _cs, _re, _ce, _wr, _wc, envmapRot):

    one_sub_wc = 1 - _wc
    one_sub_wr = 1 - _wr

    envmap_rs_cs = np.empty(shape=envmap.shape)
    envmap_rs_ce = np.empty(shape=envmap.shape)
    envmap_re_cs = np.empty(shape=envmap.shape)
    envmap_re_ce = np.empty(shape=envmap.shape)

    for r in range(0, height):
        for c in range(0, width):
            envmap_rs_cs[r, c, :] = envmap[_rs[r, c], _cs[r, c], :]
            envmap_rs_ce[r, c, :] = envmap[_rs[r, c], _ce[r, c], :]
            envmap_re_cs[r, c, :] = envmap[_re[r, c], _cs[r, c], :]
            envmap_re_ce[r, c, :] = envmap[_re[r, c], _ce[r, c], :]

    for i in range(3):
        _color1 = one_sub_wc * envmap_rs_cs[:, :, i] + _wc * envmap_rs_ce[:, :, i]
        _color2 = one_sub_wc * envmap_re_cs[:, :, i] + _wc * envmap_re_ce[:, :, i]
        envmapRot[:, :, i] = one_sub_wr * _color1 + _wr * _color2

    # for r in range(0, height):
    #     for c in range(0, width):
    #         _color1 = one_sub_wc[r, c] * envmap[_rs[r, c], _cs[r, c], :] + _wc[r, c] * envmap[_rs[r, c], _ce[r, c], :]
    #         _color2 = one_sub_wc[r, c] * envmap[_re[r, c], _cs[r, c], :] + _wc[r, c] * envmap[_re[r, c], _ce[r, c], :]
    #         color = one_sub_wr[r, c] * _color1 + _wr[r, c] * _color2
    #         envmapRot[r, c, :] = color

    return envmapRot


@timeit(PATH_LOG_GEN)
def rotateEnvmap_OPT_MATRIX(envmap, vn):
    envmapRot = np.zeros(envmap.shape, dtype=np.float32)

    up = np.array([0, 1, 0], dtype=np.float32)
    z = vn
    z = z / np.sqrt(np.sum(z * z))
    x = np.cross(up, z)
    x = x / np.sqrt(np.sum(x * x))
    y = np.cross(z, x)
    y = y / np.sqrt(np.sum(y * y))
    x, y, z = x[np.newaxis, :], y[np.newaxis, :], z[np.newaxis, :]
    R = np.concatenate([x, y, z], axis=0)
    print(R)

    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    height, width = envmap.shape[0], envmap.shape[1]

    theta = np.arange(height) / float(height - 1) * np.pi
    phi = np.arange(width) / float(width) * np.pi * 2 - np.pi
    theta = np.tile(theta.reshape(-1, 1), (1, width))
    phi = np.tile(phi.T, (height, 1))

    z = np.sin(theta) * np.cos(phi)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)

    nx = x * rx[0] + y * ry[0] + z * rz[0]
    ny = x * rx[1] + y * ry[1] + z * rz[1]
    nz = x * rx[2] + y * ry[2] + z * rz[2]

    nx = nx / (np.sqrt(1 - nz * nz) + 1e-12)
    ny = ny / (np.sqrt(1 - nz * nz) + 1e-12)

    nx = np.clip(nx, -1, 1)
    ny = np.clip(ny, -1, 1)
    nz = np.clip(nz, -1, 1)

    phiNew = np.arccos(nx)
    phiNew[ny < 0] = - phiNew[ny < 0]
    thetaNew = np.arccos(nz)

    u = (phiNew + np.pi) / 2 / np.pi
    v = 1 - thetaNew / np.pi

    _c, _r = u * (width - 1), (1 - v) * (height - 1)
    _cs, _rs = _c.astype(int), _r.astype(int)
    _ce, _re = _cs.copy() + 1, _rs.copy() + 1
    _ce[_ce > width - 1] = width - 1
    _re[_re > height - 1] = height - 1
    _wc, _wr = _c - _cs, _r - _rs

    return rotate_OPT_MATRIX(envmap, height, width, _rs, _cs, _re, _ce, _wr, _wc, envmapRot)


def rotate_OPT_MATRIX_NE(envmap, height, width, _rs, _cs, _re, _ce, _wr, _wc, envmapRot):

    one_sub_wc = 1 - _wc
    one_sub_wr = 1 - _wr

    envmap_rs_cs = np.empty(shape=envmap.shape)
    envmap_rs_ce = np.empty(shape=envmap.shape)
    envmap_re_cs = np.empty(shape=envmap.shape)
    envmap_re_ce = np.empty(shape=envmap.shape)

    for r in range(0, height):
        for c in range(0, width):
            envmap_rs_cs[r, c, :] = envmap[_rs[r, c], _cs[r, c], :]
            envmap_rs_ce[r, c, :] = envmap[_rs[r, c], _ce[r, c], :]
            envmap_re_cs[r, c, :] = envmap[_re[r, c], _cs[r, c], :]
            envmap_re_ce[r, c, :] = envmap[_re[r, c], _ce[r, c], :]

    for i in range(3):
        _color1 = one_sub_wc * envmap_rs_cs[:, :, i] + _wc * envmap_rs_ce[:, :, i]
        _color2 = one_sub_wc * envmap_re_cs[:, :, i] + _wc * envmap_re_ce[:, :, i]
        envmapRot[:, :, i] = ne.evaluate("one_sub_wr * _color1 + _wr * _color2")

    # for r in range(0, height):
    #     for c in range(0, width):
    #         _color1 = one_sub_wc[r, c] * envmap[_rs[r, c], _cs[r, c], :] + _wc[r, c] * envmap[_rs[r, c], _ce[r, c], :]
    #         _color2 = one_sub_wc[r, c] * envmap[_re[r, c], _cs[r, c], :] + _wc[r, c] * envmap[_re[r, c], _ce[r, c], :]
    #         color = one_sub_wr[r, c] * _color1 + _wr[r, c] * _color2
    #         envmapRot[r, c, :] = color

    return envmapRot


@timeit(PATH_LOG_GEN)
def rotateEnvmap_OPT_MATRIX_NE(envmap, vn):
    envmapRot = np.zeros(envmap.shape, dtype=np.float32)

    up = np.array([0, 1, 0], dtype=np.float32)
    z = vn
    z = z / np.sqrt(np.sum(z * z))
    x = np.cross(up, z)
    x = x / np.sqrt(np.sum(x * x))
    y = np.cross(z, x)
    y = y / np.sqrt(np.sum(y * y))
    x, y, z = x[np.newaxis, :], y[np.newaxis, :], z[np.newaxis, :]
    R = np.concatenate([x, y, z], axis=0)
    print(R)

    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    height, width = envmap.shape[0], envmap.shape[1]

    theta = np.arange(height) / float(height - 1) * np.pi
    phi = np.arange(width) / float(width) * np.pi * 2 - np.pi
    theta = np.tile(theta.reshape(-1, 1), (1, width))
    phi = np.tile(phi.T, (height, 1))

    z = np.sin(theta) * np.cos(phi)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)

    rx0, rx1, rx2 = rx[0], rx[1], rx[2]
    ry0, ry1, ry2 = ry[0], ry[1], ry[2]
    rz0, rz1, rz2 = rz[0], rz[1], rz[2]

    nx = ne.evaluate("x * rx0 + y * ry0 + z * rz0")
    ny = ne.evaluate("x * rx1 + y * ry1 + z * rz1")
    nz = ne.evaluate("x * rx2 + y * ry2 + z * rz2")

    nx = nx / (np.sqrt(1 - nz * nz) + 1e-12)
    ny = ny / (np.sqrt(1 - nz * nz) + 1e-12)

    nx = np.clip(nx, -1, 1)
    ny = np.clip(ny, -1, 1)
    nz = np.clip(nz, -1, 1)

    phiNew = np.arccos(nx)
    phiNew[ny < 0] = - phiNew[ny < 0]
    thetaNew = np.arccos(nz)

    pi = np.pi
    u = ne.evaluate("(phiNew + pi) / 2 / pi")
    v = ne.evaluate("1 - thetaNew / pi")

    _c, _r = u * (width - 1), (1 - v) * (height - 1)
    _cs, _rs = _c.astype(int), _r.astype(int)
    _ce, _re = _cs.copy() + 1, _rs.copy() + 1
    _ce[_ce > width - 1] = width - 1
    _re[_re > height - 1] = height - 1
    _wc, _wr = _c - _cs, _r - _rs

    return rotate_OPT_MATRIX(envmap, height, width, _rs, _cs, _re, _ce, _wr, _wc, envmapRot)


@cuda.jit()
def rotate_OPT_CUDA(envmap, height, width, _rs, _cs, _re, _ce, _wr, _wc, envmapRot):
    r, c = cuda.grid(2)
    for i in range(3):
        _color1 = (1 - _wc[r, c]) * envmap[_rs[r, c], _cs[r, c], i] + _wc[r, c] * envmap[_rs[r, c], _ce[r, c], i]
        _color2 = (1 - _wc[r, c]) * envmap[_re[r, c], _cs[r, c], i] + _wc[r, c] * envmap[_re[r, c], _ce[r, c], i]
        color = (1 - _wr[r, c]) * _color1 + _wr[r, c] * _color2
        envmapRot[r, c, i] = color


@timeit(PATH_LOG_GEN)
def rotateEnvmap_OPT_CUDA(envmap, vn):
    envmapRot = np.zeros(envmap.shape, dtype=np.float32)

    time1 = time.time()

    up = np.array([0, 1, 0], dtype=np.float32)
    z = vn
    z = z / np.sqrt(np.sum(z * z))
    x = np.cross(up, z)
    x = x / np.sqrt(np.sum(x * x))
    y = np.cross(z, x)
    y = y / np.sqrt(np.sum(y * y))
    x, y, z = x[np.newaxis, :], y[np.newaxis, :], z[np.newaxis, :]
    R = np.concatenate([x, y, z], axis=0)
    print(R)

    rx, ry, rz = R[:, 0], R[:, 1], R[:, 2]
    height, width = envmap.shape[0], envmap.shape[1]

    theta = np.arange(height) / float(height - 1) * np.pi
    phi = np.arange(width) / float(width) * np.pi * 2 - np.pi
    theta = np.tile(theta.reshape(-1, 1), (1, width))
    phi = np.tile(phi.T, (height, 1))

    z = np.sin(theta) * np.cos(phi)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)

    nx = x * rx[0] + y * ry[0] + z * rz[0]
    ny = x * rx[1] + y * ry[1] + z * rz[1]
    nz = x * rx[2] + y * ry[2] + z * rz[2]
    nx = nx / (np.sqrt(1 - nz * nz) + 1e-12)
    ny = ny / (np.sqrt(1 - nz * nz) + 1e-12)
    nx = np.clip(nx, -1, 1)
    ny = np.clip(ny, -1, 1)
    nz = np.clip(nz, -1, 1)

    phiNew = np.arccos(nx)
    phiNew[ny < 0] = - phiNew[ny < 0]
    thetaNew = np.arccos(nz)

    u = (phiNew + np.pi) / 2 / np.pi
    v = 1 - thetaNew / np.pi

    _c, _r = u * (width - 1), (1 - v) * (height - 1)
    _cs, _rs = _c.astype(int), _r.astype(int)
    _ce, _re = _cs.copy() + 1, _rs.copy() + 1
    _ce[_ce > width - 1] = width - 1
    _re[_re > height - 1] = height - 1
    _wc, _wr = _c - _cs, _r - _rs

    print("before CUDA:", time.time() - time1)

    envmap_d = cuda.to_device(envmap)

    print("before CUDA IO:", time.time() - time1)

    _rs_d, _cs_d = cuda.to_device(_rs), cuda.to_device(_cs)
    _re_d, _ce_d = cuda.to_device(_re), cuda.to_device(_ce)
    _wr_d, _wc_d = cuda.to_device(_wr), cuda.to_device(_wc)

    print("before CUDA IO:", time.time() - time1)

    height_d, width_d = cuda.to_device(envmap.shape[0]), cuda.to_device(envmap.shape[1])
    envmapRot = cuda.device_array(shape=envmap.shape, dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid_x = int(math.ceil(envmap.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(envmap.shape[1] / threads_per_block[1]))
    blocksPerGrid = (blocks_per_grid_x, blocks_per_grid_y)

    rotate_OPT_CUDA[blocksPerGrid, threads_per_block](
        envmap_d, height_d, width_d, _rs_d, _cs_d, _re_d, _ce_d, _wr_d, _wc_d, envmapRot)

    return envmapRot.copy_to_host()



