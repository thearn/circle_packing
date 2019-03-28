from openmdao.api import Problem, pyOptSparseDriver
import numpy as np
from packing import Packing
import matplotlib.pyplot as plt

np.random.seed(0)

n_obj = 100
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

p.model.add_constraint('keepout0.err_zone_dist', upper=0.0)
p.model.add_constraint('keepout1.err_zone_dist', upper=0.0)
p.model.add_constraint('space.err_space_dist', upper=0.0)
p.model.add_constraint('spheres.err_pair_dist', upper=0.0)

p.setup()

import pickle
with open('data.dat', 'rb') as f:
    data = pickle.load(f)

p['x'] = data['x']
p['y'] = data['y']
p['radius'] = data['radius']

import time
t = time.time()
for i in range(10):
    p.run_model()
    p.compute_totals()
tt = time.time() - t
print("avg. time:", tt/10)


#p.run_driver()

