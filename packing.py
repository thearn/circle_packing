import numpy as np

from spheres import Spheres
from space import Space
from keepout import KeepOut
from constraint_aggregator import ConstraintAggregator
from openmdao.api import Group, IndepVarComp


class Packing(Group):

    def initialize(self):
        self.options.declare('n_obj', types=int)
        self.options.declare('sum_constraints', default=True, types=bool)
        self.options.declare('aggregator', default='RePU', types=str)
        self.options.declare('r_space', default=1.0, types=float)
        self.options.declare('rho', default=50.0, types=float)

    def setup(self):
        n_obj = self.options['n_obj']
        agg = self.options['aggregator']
        r_space = self.options['r_space']
        rho = self.options['rho']

        inputs = IndepVarComp()
        inputs.add_output('x', val=np.zeros(n_obj))
        inputs.add_output('y', val=np.zeros(n_obj))
        inputs.add_output('radius', val=np.zeros(n_obj))

        self.add_subsystem('inputs', inputs, promotes=['*'])

        self.add_subsystem("spheres", Spheres(n_obj = n_obj,
                                              r_space = r_space),
                            promotes_inputs=['*'],
                            promotes_outputs=['area'])

        self.add_subsystem('space', Space(n_obj=n_obj,
                                          r_space=r_space),
                                    promotes_inputs=['*'])

        self.add_subsystem('keepout0', KeepOut(n_obj=n_obj,
                                              x_pos=0.1,
                                              y_pos=0.4,
                                              r_zone=0.3),
                                    promotes_inputs=['*'])

        self.add_subsystem('keepout1', KeepOut(n_obj=n_obj,
                                              x_pos=-0.2,
                                              y_pos=-0.5,
                                              r_zone=0.45),
                                    promotes_inputs=['*'])

        nc = (n_obj**2 - n_obj) // 2

        # self.add_subsystem("constr0", ConstraintAggregator(n_const=n_obj,
        #                                                aggregator=agg,
        #                                                rho=rho))
        # self.connect('keepout0.err_zone_dist', 'constr0.g')

        # self.add_subsystem("constr2", ConstraintAggregator(n_const=n_obj,
        #                                                aggregator=agg,
        #                                                rho=rho))
        # self.connect('keepout1.err_zone_dist', 'constr2.g')

        # self.add_subsystem("constr1", ConstraintAggregator(n_const=n_obj,
        #                                                aggregator=agg,
        #                                                rho=rho))

        # self.connect('space.err_space_dist', 'constr1.g')

if __name__ == '__main__':
    from openmdao.api import Problem
    import matplotlib.pyplot as plt
    np.random.seed(0)
    n_obj = 10

    """
    IMPLEMENT OPTIMIZATION PROBLEM NEXT - NEW FILE
    """


    aggregator = 'RePU'

    p = Problem()
    p.model = Packing(n_obj = n_obj, aggregator=aggregator)

    p.setup()

    p['x'] = np.random.uniform(-1, 1, n_obj)
    p['y'] = np.random.uniform(-1, 1, n_obj)
    p['radius'] = np.random.uniform(0.01, 0.3, n_obj)

    p.run_model()

    area = p['area']

    fig = plt.figure()
    ax = plt.gca()

    for i in range(n_obj):
        pos = p['x'][i], p['y'][i]
        r = p['radius'][i]
        plt.plot(pos[0], pos[1])
        circle = plt.Circle(pos, r, fill=False)
        ax.add_artist(circle)
    circle = plt.Circle((0,0), 1.0, fill=False)
    ax.add_artist(circle)

    #plt.tight_layout(pad=1)
    plt.axis('equal')
    plt.xlim(-1,1)
    plt.ylim(-1,1)

    plt.show()






