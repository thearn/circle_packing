import numpy as np
from openmdao.api import Group, Problem, ExplicitComponent
from aggregator_funcs import aggf
from rtree import index

class KeepOut(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_obj', types=int)

        self.options.declare('x_pos', default=0.0, types=float)
        self.options.declare('y_pos', default=0.0, types=float)
        self.options.declare('r_zone', default=0.5, types=float)

    def setup(self):
        n_obj = self.options['n_obj']

        self.add_input(name='x', val=np.ones(n_obj))
        self.add_input(name='y', val=np.ones(n_obj))
        self.add_input(name='radius', val=np.ones(n_obj))

        self.add_output(name='err_zone_dist', val=np.ones(n_obj))
        
        ar = np.arange(n_obj)
        self.declare_partials('err_zone_dist', ['x', 'y', 'radius'], rows=ar, cols=ar)


    def compute(self, inputs, outputs):
        n_obj = self.options['n_obj']
        x = inputs['x']
        y = inputs['y']
        r = inputs['radius']

        xp = self.options['x_pos']
        yp = self.options['y_pos']
        rp = self.options['r_zone']

        nm = np.sqrt((x - xp)**2 + (y - yp)**2)

        outputs['err_zone_dist'] = r + rp - nm

        self.de_dx = -(x - xp) / nm
        self.de_dy = -(y - yp) / nm
        self.de_dr = np.ones(n_obj)



    def compute_partials(self, inputs, partials):
        partials['err_zone_dist', 'x'] = self.de_dx
        partials['err_zone_dist', 'y'] = self.de_dy
        partials['err_zone_dist', 'radius'] = self.de_dr



if __name__ == '__main__':
    from openmdao.api import Problem, Group
    np.random.seed(1)
    n_obj = 5
    print("constraints:", (n_obj**2 - n_obj) // 2 )
    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', KeepOut(n_obj=n_obj),
                                          promotes=['*'])
    p.setup()

    p['x'] = np.random.uniform(0, 1, n_obj)
    p['y'] = np.random.uniform(0, 1, n_obj)
    p['radius'] = np.random.uniform(0, 1, n_obj)

    p.run_model()
    print(p['err_zone_dist'])

    #quit()

    p.check_partials(compact_print=True)




