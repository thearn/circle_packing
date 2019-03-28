import numpy as np
from openmdao.api import Group, Problem, ExplicitComponent
from aggregator_funcs import aggf
from rtree import index
from scipy.spatial import cKDTree

class Spheres(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_obj', types=int)
        self.options.declare('r_space', default=1.0, types=float)
        self.options.declare('aggregator', default='RePU', types=str)
        self.options.declare('rho', default=2.0, types=float)

    def setup(self):
        n_obj = self.options['n_obj']
        agg = self.options['aggregator']
        r_space = self.options['r_space']
        rho = self.options['rho']


        self.aggf = aggf[agg]
        self.rho = rho

        self.max_area = np.pi * self.options['r_space']**2

        self.add_input(name='x', val=np.ones(n_obj))
        self.add_input(name='y', val=np.ones(n_obj))
        self.add_input(name='radius', val=np.ones(n_obj))

        self.add_output(name='err_pair_dist', val=1.0)
        self.add_output(name='area', val=1.0)
 
        self.declare_partials('err_pair_dist', ['x', 'y', 'radius'])

        self.declare_partials('area', 'radius')


    def compute(self, inputs, outputs):
        self.compute_kdtree(inputs, outputs)

    def compute_kdtree(self, inputs, outputs):
        n_obj = self.options['n_obj']
        X, Y, R = inputs['x'], inputs['y'], inputs['radius']

        outputs['area'] = np.sum(np.pi * inputs['radius']**2) / self.max_area

        self.epd_x = np.zeros(n_obj)
        self.epd_y = np.zeros(n_obj)
        self.epd_r = np.zeros(n_obj)

        outputs['err_pair_dist'] = 0.0

        data = np.dstack((X, Y)).reshape(n_obj, 2)
        max_r = R.max()
        idx = cKDTree(data)

        for i in range(n_obj):
            pts = idx.query_ball_point(data[i], R[i] + max_r)
            pts = [ii for ii in pts if ii > i]

            x0,y0 = data[i]
            r0 = R[i]

            x1, y1 = data[pts].T
            r1 = R[pts]

            dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

            err, derr = self.aggf(r0 + r1 - dist, self.rho)

            self.epd_x[i] += np.sum(-(x0 - x1)/dist * derr)
            self.epd_x[pts] += (x0 - x1)/dist * derr

            self.epd_y[i] += np.sum(-(y0 - y1)/dist * derr)
            self.epd_y[pts] += (y0 - y1)/dist * derr

            self.epd_r[i] += np.sum(derr)
            self.epd_r[pts] += derr

            outputs['err_pair_dist'] += err.sum()

    def compute_rtree(self, inputs, outputs):
        n_obj = self.options['n_obj']
        X, Y, R = inputs['x'], inputs['y'], inputs['radius']
        max_r = np.max(R)

        outputs['area'] = np.sum(np.pi * inputs['radius']**2) / self.max_area

        idx = index.Index()

        for i in range(n_obj):
            idx.insert(i, [X[i], Y[i], X[i], Y[i]])

        self.epd_x = np.zeros(n_obj)
        self.epd_y = np.zeros(n_obj)
        self.epd_r = np.zeros(n_obj)

        outputs['err_pair_dist'] = 0.0

        for i in range(n_obj):

            x0, y0, r0 = X[i], Y[i], R[i]

            w = r0 + max_r
            s = idx.intersection((x0 - w, y0 - w, x0 + w, y0 + w))
            s = [j for j in list(s) if j > i]
            x1, y1, r1 = X[s], Y[s], R[s]

            dist = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

            err, derr = self.aggf(r0 + r1 - dist, self.rho)

            self.epd_x[i] += np.sum(-(x0 - x1)/dist * derr)
            self.epd_x[s] += (x0 - x1)/dist * derr

            self.epd_y[i] += np.sum(-(y0 - y1)/dist * derr)
            self.epd_y[s] += (y0 - y1)/dist * derr

            self.epd_r[i] += np.sum(derr)
            self.epd_r[s] += derr

            outputs['err_pair_dist'] += err.sum()


    def compute_(self, inputs, outputs):
        n_obj = self.options['n_obj']

        outputs['area'] = np.sum(np.pi * inputs['radius']**2) / self.max_area

        self.dt_dr = 2*np.pi*inputs['radius']
        self.epd_x = np.zeros((self.nc, n_obj))
        self.epd_y = np.zeros((self.nc, n_obj))
        self.epd_r = np.zeros((self.nc, n_obj))
        k = 0
        for i, j in combinations([i for i in range(n_obj)], 2):
            x1, y1 = inputs['x'][i], inputs['y'][i]
            x2, y2 = inputs['x'][j], inputs['y'][j]

            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < 0.05:
                dist = 0.05

            rad = inputs['radius'][i] + inputs['radius'][j]

            outputs['err_pair_dist'][k] = rad - dist

            self.epd_x[k][i] = -(x1 - x2)/dist
            self.epd_x[k][j] = (x1 - x2)/dist

            self.epd_y[k][i] = -(y1 - y2)/dist
            self.epd_y[k][j] = (y1 - y2)/dist

            self.epd_r[k][i] = self.epd_r[k][j] = 1.0

            k += 1


    def compute_partials(self, inputs, partials):
        partials['err_pair_dist', 'x'] = self.epd_x
        partials['err_pair_dist', 'y'] = self.epd_y
        partials['err_pair_dist', 'radius'] = self.epd_r

        partials['area', 'radius'] = 2*np.pi*inputs['radius']/ self.max_area


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    np.random.seed(1)
    n_obj = 5
    print("constraints:", (n_obj**2 - n_obj) // 2 )
    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', Spheres(n_obj=n_obj),
                                          promotes=['*'])
    p.setup()

    p['x'] = np.random.uniform(0, 1, n_obj)
    p['y'] = np.random.uniform(0, 1, n_obj)
    p['radius'] = np.random.uniform(0, 1, n_obj)

    p.run_model()
    print(p['err_pair_dist'])

    #quit()

    p.check_partials(compact_print=True)




