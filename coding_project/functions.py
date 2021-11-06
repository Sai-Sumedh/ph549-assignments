import numpy as np
import matplotlib.pyplot as plt
import math, time

def F(u, v, w, c1, c2, c3, U):
    '''
    shape of u, v, w: (N, N)
    c1, c2, c3, U- constants
    '''
    val = c1*v+c2*w+c3
    val[val<0] = 0 # lower threshold
    val[val>U] = U # upper threshold
    
    return val

def G(u, v, w, c4, c5, c6, V):
    val = c4*u+c5*w+c6
    val[val<0] = 0 # lower threshold
    val[val>V] = V # upper threshold
    
    return val

def H(u, v, w, c7, c8, c9, W):
    val = c7*u+c8*v+c9
    val[val<0] = 0 # lower threshold
    val[val>W] = W # upper threshold
    
    return val


def Del2(Del2x, c):
    """
    return the Laplacian of c
    d2c/dx2 + d2c/dy2
    """
    d2x_c = np.dot(Del2x, c)
    d2y_c = np.dot(Del2x, c.T).T
    Lap_c = d2x_c+d2y_c
    return Lap_c

def dudt(u, v, w, c1, c2, c3, U, cu, Du, boundary_mask, Del2x):
    reaction_term = F(u, v, w, c1, c2, c3, U)
    decay_term = -cu*u
    diffusion_term = boundary_mask*(Du*Del2(Del2x, u))
    ans = reaction_term + decay_term + diffusion_term
    return ans

def dvdt(u, v, w, c4, c5, c6, V, cv, Dv, boundary_mask, Del2x):
    reaction_term = G(u, v, w, c4, c5, c6, V)
    decay_term = -cv*v
    diffusion_term = boundary_mask*(Dv*Del2(Del2x, v))
    ans = reaction_term + decay_term + diffusion_term
    return ans


def dwdt(u, v, w, c7, c8, c9, W, cw, Dw, boundary_mask, Del2x):
    reaction_term = H(u, v, w, c7, c8, c9, W)
    decay_term = -cw*w
    diffusion_term = boundary_mask*(Dw*Del2(Del2x, w))
    ans = reaction_term + decay_term + diffusion_term
    return ans

