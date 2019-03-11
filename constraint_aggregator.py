import numpy as np
from aggregator_funcs import aggf, transform
from openmdao.api import Group, Problem, ExplicitComponent

class ConstraintAggregator(ExplicitComponent):
    """
    Transform and aggregate constraints to a single value.
    """
    def initialize(self):
        self.options.declare('n_const', types=int)
        self.options.declare('aggregator', types=str)
        self.options.declare('rho', default=50.0, types=float)
        self.options.declare('reversed', default=False, types=bool)

    def setup(self):
        nc = self.options['n_const']
        agg = self.options['aggregator']
        self.reversed = False
        if self.options['reversed']:
            self.reversed = True
        self.aggf = aggf[agg]

        self.add_input(name='g', val=np.ones(nc))
        self.add_output(name='c', val=1.0)

        self.declare_partials('c', 'g')

    def compute(self, inputs, outputs):
        rho = self.options['rho']
        g = inputs['g']
        scale = 1.0
        if self.reversed:
            scale = -1.0
        k, dk = self.aggf(scale*g, rho)

        outputs['c'] = np.sum(scale*k)
        self.dk = dk

    def compute_partials(self, inputs, partials):
        partials['c', 'g'] = self.dk

class ConstraintAggregatorVec(ExplicitComponent):
    """
    Transforms constraints, but doesn't sum.
    With aggregator=None, just passes through.
    """
    def initialize(self):
        self.options.declare('n_const', types=int)
        self.options.declare('aggregator', types=str)
        self.options.declare('rho', default=50.0, types=float)

    def setup(self):
        nc = self.options['n_const']
        agg = self.options['aggregator']
        self.aggf = aggf[agg]

        self.add_input(name='g', val=np.ones(nc))
        self.add_output(name='c', val=np.ones(nc))

        ar = np.arange(nc)
        self.declare_partials('c', 'g', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        rho = self.options['rho']
        g = inputs['g']
        k, dk = self.aggf(g, rho)
        outputs['c'] = k
        self.dk = dk

    def compute_partials(self, inputs, partials):
        partials['c', 'g'] = self.dk


if __name__ == '__main__':
    from openmdao.api import Problem, Group
    np.random.seed(1)
    nc = 20
    m = 0.0

    p = Problem()
    p.model = Group()
    p.model.add_subsystem('test', ConstraintAggregator(n_const=nc,
                                          reversed=True,
                                          rho=2.0,
                                          aggregator='RePU'), promotes=['*'])
    p.setup()
    g = np.linspace(-2,10,nc)
    p['g'] = g

    p.run_model()

    print(p['c'])

    p.check_partials(compact_print=True)