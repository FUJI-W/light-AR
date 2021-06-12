import os
import time
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import cv2


def timeit(logfile='out.log'):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _t0 = time.time()
            res = func(*args, **kwargs)
            _t1 = time.time()
            with open(logfile, 'a') as opened_file:
                opened_file.write("%s: %s seconds\n" % (func.__name__, str(_t1 - _t0)))
            return res

        return wrapper

    return decorate


def chartit(logfile='out.log'):
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
    x = list(means.values())
    y = list(means.keys())
    x.reverse()
    y.reverse()

    plt.axes([0.125, 0.1, 0.8, 0.8])
    plt.barh(range(len(x)), x, tick_label=y, facecolor='#9999ff', edgecolor='white')
    plt.xlabel('time/s')
    plt.title('Average Running Time of Each Function')
    for a, b in enumerate(x):
        plt.text(b + 0.1, a, '%.6s s' % b, va='center')

    plt.axes([0.65, 0.5, 0.25, 0.25])
    p_means = dict()
    for _m in means:
        if means.get(_m) > 0.1:
            p_means[_m] = means.get(_m)
        else:
            if 'others' not in means:
                p_means['others'] = 0.0
            p_means['others'] = p_means['others'] + means.get(_m)

    px = np.asarray(list(p_means.values()))
    px = px / np.sum(px)
    py = list(p_means.keys())

    plt.pie(px, labels=py,
            autopct=lambda _x: ('%.2f%%' % _x) if _x > 0.001 else '<0.01%',
            pctdistance=0.5)
    plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, color='white'))

    fig = plt.gcf()
    fig.set_size_inches((16, 9), forward=False)
    fig.savefig(os.path.join(os.path.dirname(logfile), 'out.log.png'), dpi=120)

    # plt.show()
    return fig


def showit(rgbe_path='scene.rgbe'):
    _hdr1 = cv2.imread(rgbe_path, cv2.IMREAD_ANYDEPTH)
    cv2.imshow("", _hdr1.astype(np.float64))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
