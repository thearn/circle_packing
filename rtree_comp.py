from rtree import index
from scipy.spatial import cKDTree
import numpy as np
import time
def distance(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def RTree(x, y, radius, r_space):
    idx = index.Index()

    n_obj = x.size

    nc = (n_obj**2 - n_obj) // 2

    err_distance = []

    for i in range(n_obj):
        idx.insert(i, [x[i], y[i], x[i], y[i]])

    max_r = np.max(radius)
    for ii in range(n_obj):

        x0, y0, r0 = x[ii], y[ii], radius[ii]

        w = r0 + max_r
        s = idx.intersection((x0 - w, y0 - w, x0 + w, y0 + w))

        for i in s:
            if i <= ii:
                continue
            x1, y1 = x[i], y[i]
            dist = distance(x0, x1, y0, y1)
            err = r0 + radius[i] - dist
            if err > 0:
                dx0 = -(x0 - x1)/dist
                dx1 = (x0 - x1)/dist
                dy0 = -(y0 - y1)/dist
                dy1 = (y0 - y1)/dist
                err_distance.append([ii, i, err, dx0, dx1, dy0, dy1])

    return err_distance

def KDTree(x, y, radius, r_space):
    n = x.size
    data = np.dstack((x, y)).reshape(n, 2)
    max_r = radius.max()
    idx = cKDTree(data)

    err_distance = []
    for i in range(n):
        pts = idx.query_ball_point(data[i], radius[i] + max_r)

        x0,y0 = data[i]
        r0 = radius[i]

        for n_idx in pts:
            if n_idx <= i:
                continue
            x1, y1 = data[n_idx]
            r1 = radius[n_idx]
            dist = distance(x0, x1, y0, y1)
            err = r0 + r1 - dist
            if err > 0:
                dx0 = -(x0 - x1)/dist
                dx1 = (x0 - x1)/dist
                dy0 = -(y0 - y1)/dist
                dy1 = (y0 - y1)/dist
                err_distance.append([i, n_idx, err, dx0, dx1, dy0, dy1])

    return err_distance



if __name__ == '__main__':
    import time
    np.random.seed(0)
    r_space = 1.0

    n_obj = 500

    x = np.random.uniform(-r_space, r_space, n_obj)
    y = np.random.uniform(-r_space, r_space, n_obj)
    radius = np.random.uniform(0.01, 0.25, n_obj)

    
    t = time.time()
    ed = KDTree(x, y, radius, r_space)
    print("t:", time.time() - t)
    print(sum([d[0] for d in ed]))

    t = time.time()
    ed = RTree(x, y, radius, r_space)
    print("t:", time.time() - t)
    print(sum([d[0] for d in ed]))


