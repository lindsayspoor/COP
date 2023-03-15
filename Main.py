from Molecular_Dynamics import Molecular_Dynamics


dynamics = Molecular_Dynamics(dim=3, N=2,N_unitcells=3, L=20, mass=1, n_timesteps=300, h=0.01, sigma=1, epsilon=1,T=1, k_b=1,rho=0.7, correction_i=10,periodic='True', update_method='verlet', lattice='True', plot='True', random_seed=111)

dynamics.run_simulation()
