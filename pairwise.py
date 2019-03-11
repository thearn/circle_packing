import numpy as np
from itertools import combinations
pi = np.pi


def pairwise(x, y, radius, n_obj, r_space):
    area = np.sum(pi * radius**2) / (pi * r_space**2)
    nc = (n_obj**2 - n_obj) // 2
    dt_dr = 2 * pi * radius
    epd_x = np.zeros((nc, n_obj))
    epd_y = np.zeros((nc, n_obj))
    epd_r = np.zeros((nc, n_obj))
    err_pair_dist = np.zeros(nc)
    k = 0
    for i, j in combinations([i for i in range(n_obj)], 2):
        x1, y1 = x[i], y[i]
        x2, y2 = x[j], y[j]

        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if dist < 0.05:
            dist = 0.05
        rad = radius[i] + radius[j]

        err_pair_dist[k] = rad - dist

        epd_x[k][i] = -(x1 - x2)/dist
        epd_x[k][j] = (x1 - x2)/dist

        epd_y[k][i] = -(y1 - y2)/dist
        epd_y[k][j] = (y1 - y2)/dist

        epd_r[k][i] = epd_r[k][j] = 1.0

        k += 1
    return area, err_pair_dist, dt_dr, epd_x, epd_y, epd_r

if __name__ == '__main__':
    import time
    from pw import pairwise as pw2
    np.random.seed(0)
    n_obj = 3
    r_space = 1.0
    # 4.016615051019934
    x = np.random.uniform(0, 1, n_obj)
    y = np.random.uniform(0, 1, n_obj)
    r = np.random.uniform(0, 1, n_obj)

    t = time.time()
    e = pairwise(x, y, r, n_obj, r_space)
    print("time python:", time.time() - t)
    print("result:")
    for ee in e:
        print(np.sum(ee))


    print()
    t = time.time()
    e = pw2(x, y, r, n_obj, r_space)
    print("time cython:", time.time() - t)
    print("result:")
    for ee in e:
        print(np.sum(ee))
