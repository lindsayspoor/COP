### This file contains the plotting functions necessary for the simulation and evaluation of the Molecular_Dynamics ###

import matplotlib.pyplot as plt
import numpy as np

def plot_particles(self):
  '''Plotting function to plot the particle trajectories.'''

  if self.dim==2:
    plt.figure(figsize=(10,10))

    for i in range(self.N):
      plt.plot(self.positions[:,i,0], self.positions[:,i,1], label=f"particle {i}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Particle trajectories")
    plt.show()

  if self.dim==3:

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    for i in range(self.N):
      ax.plot(self.positions[:,i,0], self.positions[:,i,1], self.positions[:,i,2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Particle trajectories")
    plt.savefig(f"trajectories_update_{self.update_method}_N_{self.N}_n_timesteps_{self.n_timesteps}_fcc_{self.lattice}_h_{self.h}_rho_{self.rho}.pdf")
    plt.show()


  print("plot_particles() is done.")
  