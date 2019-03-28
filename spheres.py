import numpy as np
from openmdao.api import Group, Problem, ExplicitComponent
from aggregator_funcs import aggf
from rtree import index
from scipy.spatial import cKDTree
from itertools import combinations

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

        self.nc = (n_obj**2 - n_obj) // 2

        self.aggf = aggf[agg]
        self.rho = rho

        self.max_area = np.pi * self.options['r_space']**2

        self.add_input(name='x', val=np.ones(n_obj))
        self.add_input(name='y', val=np.ones(n_obj))
        self.add_input(name='radius', val=np.ones(n_obj))

        self.add_output(name='err_pair_dist', val=np.zeros((n_obj, n_obj)))
        self.add_output(name='area', val=1.0)
 
        self.declare_partials('err_pair_dist', ['x', 'y', 'radius'])

        self.declare_partials('area', 'radius')

    def compute(self, inputs, outputs):
        n_obj = self.options['n_obj']

        outputs['area'] = np.sum(np.pi * inputs['radius']**2) / self.max_area

        self.dt_dr = 2*np.pi*inputs['radius']
        self.epd_x = np.zeros((n_obj, n_obj, n_obj))
        self.epd_y = np.zeros((n_obj, n_obj, n_obj))
        self.epd_r = np.zeros((n_obj, n_obj, n_obj))

        for i, j in combinations([i for i in range(n_obj)], 2):
            x1, y1 = inputs['x'][i], inputs['y'][i]
            x2, y2 = inputs['x'][j], inputs['y'][j]

            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < 0.05:
                dist = 0.05

            rad = inputs['radius'][i] + inputs['radius'][j]

            outputs['err_pair_dist'][i][j] = outputs['err_pair_dist'][j][i] = rad - dist

            self.epd_x[i, j, i] = self.epd_x[j, i, i] = -(x1 - x2)/dist
            self.epd_x[i, j, j] = self.epd_x[j, i, j] = (x1 - x2)/dist

            self.epd_y[i, j, i] = self.epd_y[j, i, i] = -(y1 - y2)/dist
            self.epd_y[i, j, j] = self.epd_y[j, i, j] = (y1 - y2)/dist

            self.epd_r[i, j, i] = self.epd_r[j, i, i] = 1.0
            self.epd_r[i, j, j] = self.epd_r[j, i, j] = 1.0


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




