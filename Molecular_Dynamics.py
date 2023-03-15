#imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist as dist

class Molecular_Dynamics:
  def __init__(self, dim=2, N=2, N_unitcells=3, L=20, mass=1, n_timesteps=300, h=0.01, sigma=1, epsilon=1, rho=0.7, periodic='True', update_method='verlet', lattice='True', plot='True'):
    '''This class defines the simulation of molecular particles and its dynamics.'''

    self.dim = dim
    self.N = N
    self.N_unitcells = N_unitcells
    self.L = L
    self.mass= mass
    self.timelength = 100 #s
    self.n_timesteps = n_timesteps
    self.h = h#self.timelength/self.n_timesteps
    self.sigma=sigma
    self.epsilon=epsilon
    self.rho=rho
    self.periodic=periodic
    self.update_method=update_method
    self.lattice = lattice
    self.plot=plot




  def initialize_positions(self):
    '''Initialize positions of all particles.'''
    
    print("initialize_positions()")
    
    if self.lattice=='False':
      if self.N==2:
        self.positions=np.zeros((self.n_timesteps, self.N, self.dim))
        self.positions[0,0,0] = 0.3*self.L
        self.positions[0,0,1] = 0.7*self.L
        self.positions[0,1,0] = 0.5*self.L
        self.positions[0,1,1] = 0.69*self.L

      else:
        self.positions=np.zeros((self.n_timesteps, self.N, self.dim))
        self.positions[0,:,:] = np.random.random_sample((self.N,self.dim))*self.L


    if self.lattice=='True':
      print("initialize_fcc_lattice()")
      self.initialize_fcc_lattice()


    print("initialize_positions() is done.")


  def initialize_velocities(self):
    '''Initialize velocities of all particles.'''
   
    if self.lattice=='False':
      self.velocities = np.zeros((self.n_timesteps, self.N, self.dim))

      if self.N==2:
        self.velocities[0,0,0] = 1.2
        self.velocities[0,1,0] = -1.2
      
      else:
        self.velocities[0,:,:] = np.random.normal(loc=0, scale=self.sigma, size=(self.N, self.dim))  #according to MB-distribution this follows a gaussian

    if self.lattice=='True':
      self.velocities = np.zeros((self.n_timesteps, self.N, self.dim))
      self.velocities[0,:,:] = np.random.normal(loc=0, scale=self.sigma, size=(self.N, self.dim)) #moeten we nog fixen
      
    print("initialize_velocities() is done.")




  def initialize_fcc_lattice(self):
    '''Initializes the fcc lattice.'''

    #self.lc, self.N = self.lattice_constant()
    self.N = 4*(self.N_unitcells**3)

    self.L = (self.N*self.mass/self.rho)**(1/3)

    self.initial_fcc = np.array([[0,0,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0]])

    self.positions=np.zeros((self.n_timesteps, self.N, self.dim))

    k=0
    for i in range(self.N_unitcells):
      for j in range(self.N_unitcells):
        self.positions[0,k:k+4,:] = self.initial_fcc + np.array([i,j,0]) #to set the 'ground level'of the lattice
        k+=4 #for the next level again, 4 atoms
        for m in range(self.N_unitcells-1): #-1 because we already have ground level at this point
          self.positions[0,k:k+4,:] = self.initial_fcc + np.array([i,j,m+1])
          k+=4

    self.positions = self.L/self.N_unitcells * self.positions
    print("initialize_fcc_lattice() is done.")



  def plot_initial_fcc_particles(self):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(self.positions[0,:,0], self.positions[0,:,1], self.positions[0,:,2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Initial positions fcc")
    plt.show()



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
      plt.savefig(f"trajectories_update_{update_method}_N_{self.N}_n_timesteps_{self.n_timesteps}_fcc_{self.lattice}_h_{self.h}_rho_{self.rho}.pdf")
      plt.show()


    print("plot_particles() is done.")


  def force_contribution(self,n,i,j):
    '''Computes the contribution of the force for the n-th timestep and the i-th particle with respect to the j-th particle.'''


    r = self.minimum_image_conv(n,i,j) #r is the direction of the particle with respect to the other particles
    norm_r = np.linalg.norm(r) #the norm or magnitude of r
    r_norm = r/norm_r #the normalized direction of the particle with respect to the other particles


    #store for the n-th timestep
    self.r_array[n,:] = r  
    self.norm_r_array[n,i,j] = norm_r 
    #self.r_norm_array[n,:] = r_norm


    dU_dr = 4*self.epsilon*(-12*self.sigma**12*norm_r**(-13)+6*self.sigma**6*norm_r**(-7))

    force_contribution = -dU_dr * r_norm

    return force_contribution


  def forces(self,n, i):
    '''Computes the total force of the n-th timestep for all particles in all directions.'''

    force = np.zeros(shape=self.dim)
    for j in range(self.N):
      if j==i:  #kunnen we deze if-statement er niet uithalen, en dan de np.where(np.isnan()) gebruiken om die op 0 te zetten wanneer i==j?
        continue
      force+= self.force_contribution(n, i, j)
      
    return force


  def minimum_image_conv(self,n,i,j):
    '''Makes sure the minimum image convention holds for periodic boundary conditions.'''

    r = (self.positions[n,i,:] - self.positions[n,j,:] + self.L/2) % self.L - self.L/2

    return r
 


  def update_euler(self,n):
    '''Updates position and velocity via the Euler rule.'''

    forces = np.array([self.forces(n,i) for i in range(self.N)])


    self.positions[n+1,:,:] = self.positions[n,:,:] + self.velocities[n,:,:]*self.h
    self.positions[n+1,:,:] = self.positions[n+1,:,:] % self.L #if the 'new' position is outside of the box then the partice re-enters the box on the other side

    self.velocities[n+1,:,:] = self.velocities[n,:,:] + forces*(self.h/self.mass)
    
    print("update_euler() is done.")




  def update_verlet(self,n):
    '''Updates position and velocity via the verlet algorithm.'''

    #x(t+h)=x(t)+v(t)*h+(h**2/2*mass)*F(x(t))+O(h**3)
    forces_t = np.array([self.forces(n,i) for i in range(self.N)])
    self.forces_array[n,:,:] = forces_t
    
    #print("updating positions[n+1]")
    self.positions[n+1,:,:] = self.positions[n,:,:] + self.velocities[n,:,:]*self.h + (self.h**2/(2*self.mass))*forces_t
    self.positions[n+1,:,:] = self.positions[n+1,:,:] % self.L #if the 'new' position is outside of the box then the partice re-enters the box on the other side
    

    forces_t_h = np.array([self.forces(n+1,k) for k in range(self.N)])
    self.forces_array[n+1,:,:] = forces_t_h

    #v(t+h)=v(t)+(h/2*mass)*F(x(t+h))+F(x(t))

    self.velocities[n+1,:,:] = self.velocities[n,:,:]+(self.h/(2*self.mass))*(forces_t_h + forces_t)
    
    #print("update_verlet() is done.")


  def kinetic_energy(self):
    '''Computes the kinetic energy of the entire system for all timesteps.'''

    E_kin_total=[]
    for n in range(self.n_timesteps):
      E_kin_all=[]
      for j in range(self.N):
        E_kin_particle = 0.5*np.linalg.norm(self.velocities[n,j,:])**2 
        E_kin_all.append(E_kin_particle)
      E_kin_total.append(E_kin_all)

    E_kin_total=np.array(E_kin_total)

    self.E_kin=np.sum(E_kin_total, axis=1)
    
    print("kinetic_energy() is done.")




  def potential_energy(self):
    '''Computes the potential energy of the entire system for all timesteps.'''
    
    print(self.norm_r_array)
    
    E_pot_total=[]
    for i in range(self.n_timesteps):
      E_pot_all=[]
      for j in range(self.N):
        E_pot_particle = np.sum(np.where(np.isnan(0.5*self.lennard_jones(self.norm_r_array[i,j,:])),0,0.5*self.lennard_jones(self.norm_r_array[i,j,:])))
        E_pot_all.append(E_pot_particle)
      E_pot_total.append(E_pot_all)

    E_pot_total=np.array(E_pot_total)

    self.E_pot=np.sum(E_pot_total, axis=1)

    
    print("potential_energy() is done.")






  def lennard_jones(self,r):
    '''Computes the Lennard-Jones potential.'''

    self.U = 4*self.epsilon*((self.sigma/r)**12-(self.sigma/r)**6)

    return self.U


  def plot_energies(self):
    '''Plotting function to plot the energy of entire the system for all given timesteps.'''


    plt.figure(figsize=(10,8))
    plt.plot(np.arange(self.n_timesteps),self.E_kin, label="E_kin")
    plt.plot(np.arange(self.n_timesteps), self.E_pot, label="E_pot")
    plt.plot(np.arange(self.n_timesteps), self.E_kin+self.E_pot, label="E_total")
    plt.xlabel("timesteps")
    plt.ylabel("E")
    plt.legend()
    plt.title("Energy diagram")
    plt.savefig(f"energy_update_{update_method}_N_{self.N}_n_timesteps_{self.n_timesteps}_fcc_{self.lattice}_h_{self.h}_rho_{self.rho}.pdf")
    plt.show()
    
    print("plot_energies() is done.")



  def run_simulation(self):
    '''Runs the simulation for the molecular dynamics.'''


    #initialize positions and velocities of all particles
    print("initialize positions...")
    self.initialize_positions()



    #self.plot_initial_fcc_particles()


    print('initialize velocities...')
    self.initialize_velocities()
    
    self.r_array = np.zeros((self.n_timesteps, self.dim))
    self.norm_r_array = np.zeros((self.n_timesteps, self.N, self.N))
    self.forces_array = np.zeros((self.n_timesteps, self.N, self.dim))

    #update the positions and velocities of all particles
    print("loop trough all timesteps...")
    for n in range(self.n_timesteps-1):
      if self.update_method == 'euler':
        print(f"update for timestep {n}")
        self.update_euler(n)
      if self.update_method == 'verlet':
        print(f"update for timestep {n}")
        self.update_verlet(n)
        
    print("compute kinetic energy...")
    self.kinetic_energy()
    print("compute potential energy...")
    self.potential_energy()
    
    
    
    if self.plot:
    
      #plot the trajectories
      print("plot the trajectories")
      self.plot_particles()



      #plot the energies
      print("plot the energies")
      self.plot_energies()
  
  print("simulation is done.")



if __name__ == '__main__':
    run_simulation()