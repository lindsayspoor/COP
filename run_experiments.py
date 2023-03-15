### This file contains experiments on the h-value test, ... ###

import numpy as np
import matplotlib.pyplot as plt
from Molecular_Dynamics import Molecular_Dynamics


 

### h-value test ###


def h_value_test(h_list, seeds):
  
  E_difference_array =[]

  for h in h_list:
  
    E_difference_list = []
  
    for seed in seeds:
  
      dynamics = Molecular_Dynamics(dim=3, N=2,N_unitcells=3, L=20, mass=1, n_timesteps=500, h=h, sigma=1, epsilon=1,T=1, k_b=1,rho=0.7, correction_i=10,periodic='True', update_method='verlet', lattice='True', plot='False', random_seed=seed)

      dynamics.run_simulation()
    
      E_difference = np.max(dynamics.E_total)-np.min(dynamics.E_total)
      E_difference_list.append(E_difference)
  
    E_difference_array.append(E_difference_list)
  
  E_difference_avg = np.mean(E_difference_array, axis=1)
  
  h_best_index=np.argwhere(E_difference_avg == np.min(E_difference_avg))[0,0]
  print(h_best_index)
  best_h = h_list[h_best_index]
  
  plt.figure(figsize=(10,10))
  plt.plot(h_list, E_difference_avg)
  plt.scatter(best_h, E_difference_avg[h_best_index], color='red', label='best_h')
  plt.xlabel("h-value")
  plt.xticks(h_list)
  plt.ylabel("Energy difference")
  plt.title("Energy difference for different h settings")
  plt.yscale('log')
  plt.legend()
  plt.savefig("h_test.pdf")
  plt.show()
  
  
  
  return best_h


h_list = [0.5,0.1,0.05,0.01,0.001,0.0001]
seeds = [4250,1376,2098,4567]

best_h = h_value_test(h_list, seeds)

print(f"The best value for h is {best_h}")




    
    
    
    








'''
run_experiment script 
eerste experiment 

h_test for many time steps
h_list=[1.,0.5,0.2,0.1,0.01]
seeds=[4250,1376,2098,4567,3215]
for h in h_list:
for seed in seeds:
run blabla **include seed in argument**
average over the 5 seeds

fig idea: plot Emax-Emin goal: 0 then it's conserved giving optimal value for h 
(pick h with minimal E diff)

'''


'''
dynamics = Molecular_Dynamics(dim=3, N=2,N_unitcells=3, L=20, mass=1, n_timesteps=300, h=0.01, sigma=1, epsilon=1,T=1, k_b=1,rho=0.7, correction_i=10,periodic='True', update_method='verlet', lattice='True', plot='True')



dynamics.run_simulation()

'''