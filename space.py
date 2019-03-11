import numpy as np
from openmdao.api import Group, Problem, ExplicitComponent

from itertools import combinations

class Space(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_obj', types=int)
        self.options.declare('r_space', default=1.0, types=float)

    def setup(self):
        n_obj = self.options['n_obj']

        self.add_input(name='x', val=np.zeros(n_obj))
        self.add_input(name='y', val=np.zeros(n_obj))

        self.add_input(name='radius', val=np.ones(n_obj))

        self.add_output(name='err_space_dist', val=np.ones(n_obj))

        ar = np.arange(n_obj)
        self.declare_partials('err_space_dist', ['radius', 'x', 'y'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        n_obj = self.options['n_obj']
        r_s = self.options['r_space']

        x = inputs['x']
        y = inputs['y']
        r = inputs['radius']

        nm = np.sqrt(x**2 + y**2)

        outputs['err_space_dist'] = (nm + r) - r_s

    def compute_partials(self, inputs, partials):
        n_obj = self.options['n_obj']
        r_s = self.options['r_space']

        x = inputs['x']
        y = inputs['y']
        r = inputs['radius']

        nm = np.sqrt(x**2 + y**2)

        partials['err_space_dist', 'radius'] = np.ones(n_obj)
        partials['err_space_dist', 'x'] = x/nm
        partials['err_space_dist', 'y'] = y/nm


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    np.random.seed(1)
    n_obj = 10
    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', Space(n_obj=n_obj),
                                          promotes=['*'])
    p.setup()

    p['x'] = np.random.uniform(0, 1, n_obj)
    p['y'] = np.random.uniform(0, 1, n_obj)
    p['radius'] = np.random.uniform(0, 1, n_obj)

    p.run_model()
    print(p['err_space_dist'])

    #quit()

    p.check_partials(compact_print=True)




