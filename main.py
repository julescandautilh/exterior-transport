### third party libraries
import numpy as np
import matplotlib.colors as mcolors

### custom functions
from functions import display_timelapse, launch_simulation, make_gif
from functions import display_figure

### grid and algorithm parameters

m = np.pi
N = 256
L = 4
h = L/N
eps = 2*h
eps_transport = 1e-2
time_step = eps/2
nb_iter = 10
c = 5


### shape parameters

shape = 'annulus'
extra = {'rmin':0.1}


### main of main

U_0, U_1, list_of_E, dic_param, list_of_U = launch_simulation(m, N, L, h, eps, 
                        eps_transport, time_step, nb_iter, c, 
                        shape=shape, extra=extra, timelapse=True)

colors = ['#ffffff', '#ffb3b3', '#b3b3ff']
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)

fig = display_timelapse(U_0, U_1, list_of_E, dic_param, cmap, list_of_U)