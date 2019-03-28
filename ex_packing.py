from openmdao.api import Problem, pyOptSparseDriver
import numpy as np
from packing import Packing
import matplotlib.pyplot as plt

np.random.seed(0)

n_obj = 700
aggregator = 'RePU'
r_space = 1.0
sum_constraints = True

params = {'RePU' : 2.0,
          'ReEU' : 0.24,
          'Sigmoid' : 1000.0,
          'SigmoidSq' : 1000.0,
          'Erf' : 1000.0,
          'KS' : 500.0,
          'None' : 0.0}


n_c = n_obj + (n_obj**2 - n_obj) // 2
p = Problem(model=Packing(n_obj=n_obj,
                          aggregator=aggregator,
                          rho=params[aggregator],
                          r_space=r_space,
                          sum_constraints=sum_constraints))

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = True
p.driver.opt_settings['iSumm'] = 6

p.driver.opt_settings["Major step limit"] = 2.0 #2.0
p.driver.opt_settings['Major iterations limit'] = 1000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-4
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-4

#p.driver.opt_settings['Minor iterations limit'] = 10000
#p.driver.opt_settings['Minor feasibility tolerance'] = 1.0e-4

# p.driver.opt_settings['QPSolver'] = 'QN' #Cholesky, QN, CG
# p.driver.opt_settings['Partial price'] = 1 # 1 to 10

p.model.add_objective('area', scaler=-1.0) # maximize area

p.model.add_design_var('x', lower=-r_space, upper=r_space)
p.model.add_design_var('y', lower=-r_space, upper=r_space)

uniform_r = r_space / np.sqrt(n_obj)
p.model.add_design_var('radius', lower=uniform_r/4, upper=1.0)

p.model.add_constraint('constr0.c', upper=0.0)
p.model.add_constraint('constr1.c', upper=0.0)
p.model.add_constraint('constr2.c', upper=0.0)
p.model.add_constraint('spheres.err_pair_dist', upper=0.0)

p.setup()




p['x'] = np.random.uniform(-r_space, r_space, n_obj)
p['y'] = np.random.uniform(-r_space, r_space, n_obj)
p['radius'] = 1e-9 * np.ones(n_obj) # 0.1 * np.ones(n_obj)

p.run_model()

p.run_driver()

area = p['area']

fig = plt.figure()
plt.title('n = %d, np = %d, nc = %d, area = %0.1f%%' % (n_obj, 3*n_obj, n_c,
                                                        100*area))
ax = plt.gca()

max_r = p['radius'].max()
min_r = p['radius'].min()
for i in range(n_obj):
    pos = p['x'][i], p['y'][i]
    r = p['radius'][i]
    col = plt.cm.copper((r - min_r) / (max_r - min_r))
    plt.plot(pos[0], pos[1])
    circle = plt.Circle(pos, r, fill=True, color=col, alpha=0.57)
    ax.add_artist(circle)
    circle = plt.Circle(pos, r, fill=False, linewidth=0.5)
    ax.add_artist(circle)
circle = plt.Circle((-0.2, -0.5), 0.45, fill=False, hatch='////')
ax.add_artist(circle)
circle = plt.Circle((0.1, 0.4), 0.3, fill=False, hatch='////')
ax.add_artist(circle)
circle = plt.Circle((0, 0), r_space, fill=False)
ax.add_artist(circle)

#plt.tight_layout(pad=1)
plt.axis('equal')
plt.xlim(-1.5*r_space, 1.5*r_space)
plt.ylim(-1.5*r_space, 1.5*r_space)

plt.show()


