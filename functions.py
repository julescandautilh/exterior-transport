
# standard libraries
import time
import os

# third party libraries
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw
from scipy import signal
from io import BytesIO

# implementation of the secant method on the function f
def secante(f, x0,x1,max_iter,tol):

    i = 0
    if x0 == x1:
        raise Exception("x0 and x1 are equal")
    
    pente = (f(x1)-f(x0))/(x1-x0)
    x_pp,x_p = x0,x1
    x = x1-f(x1)/pente
    while i < max_iter and np.abs(x_p-x) > tol:
        x_p,x_pp = x,x_p
        pente = (f(x_p)-f(x_pp))/(x_p-x_pp)
        x = x-f(x)/pente
        i += 1

    return (x,i)


def uniform_kernel(radius):

    size = 2*radius+1
    value = 1.0/size**2

    return np.full((size,size),value)

    
def gauss_kernel(sigma, radius):
    
    size = int(2*radius+1)
    t = sigma**2
    
    kernel=np.exp(-np.array([[((i-radius)**2+(j-radius)**2)/(2*t) 
                              for i in range(size)] for j in range(size)],
                                dtype=float))
    kernel = kernel/kernel.sum()
    
    return kernel


# builds an NxN array where the mass takes the form of the given shape
def build_source(stype, N, L, mass, extra=None, blur_kernel=gauss_kernel(2,2)):
    
    h = L/N
    img=Image.new("F",(N,N),0)
    draw = ImageDraw.Draw(img)
    cx,cy=N/2,N/2
    adjust_mass = False
    
    if stype=="disk":
        radius = np.sqrt(mass/np.pi)/h
        draw.ellipse([(cx-radius, cy-radius), 
                      (cx+radius, cy+radius)], fill=1)

    elif stype=='ellipse':
        if extra is None:
            ratio = 2
        else:
            ratio = extra['ratio']
        b = np.sqrt(mass/(ratio*np.pi))/h
        a = b*ratio
        draw.ellipse([(cx-a, cy-b), 
                      (cx+a, cy+b)],fill=1)

    elif stype=="disk_with_cusp":
        scaling = 1
        draw.ellipse([(N/2-N/4, N/2-N/4), 
                      (N/2+N/4, N/2+N/4)],fill=1)
        draw.polygon([(N/2-N/8,N/2), (N/2+N/8,N/2),(N/2,15*N/16)],fill=1)
        adjust_mass = True

    elif stype=="rectangle":
        if extra is None:
            ratio = 2
        else:
            ratio = extra['ratio']
        l = np.sqrt(mass/ratio)/h
        L = l*ratio
        cx1,cy1 = (N-L)/2,(N-l)/2
        cx2,cy2 = N-cx1, N-cy1
        draw.rectangle([(cx1,cy1),(cx2,cy2)], fill=1)

    elif stype=="rectangle_high":
        draw.rectangle([(N/4,N*(1.+3./10)/2),(3*N/4,N*(1.+7./10)/2)], 
                       fill=1)
        adjust_mass = True

    elif stype=="square":
        extra['ratio'] = 1
        return build_source("rectangle",N,L,mass,extra=extra,
                            blur_kernel=blur_kernel)
    elif stype=="annulus":
        if extra is None:
            rmin = 1.
        else:
            rmin = extra['rmin']
        rmax = np.sqrt(mass/np.pi+rmin**2)
        rmin,rmax = rmin/h,rmax/h
        draw.ellipse([(cx-rmax, cx-rmax), 
                      (cx+rmax, cx+rmax)],fill=1)
        draw.ellipse([(cx-rmin, cx-rmin), 
                      (cx+rmin, cx+rmin)],fill=0)

    elif stype=='annulus_with_cusp':
        draw.ellipse([(N/4,N/4),(3*N/4.,3*N/4)],fill=1)
        draw.ellipse([(3*N/8,3*N/8),(5*N/8.,5*N/8)],fill=0)
        draw.polygon([(N/2+N/16, N/4+N/16), (3*N/4, N/4), 
                      (3*N/4-N/16, N/2-N/16)],fill=1)
        adjust_mass = True

    elif stype=="half_disk":
        draw.chord([(N/4,N*(1.-1./2)/2),(3*N/4,N*(1.+1./2)/2)],
                   180,360,fill=1)
        adjust_mass = True

    elif stype=="triangle":
        if extra is None:
            ratio = 2
        else:
            ratio = extra['ratio']
        B = np.sqrt(2*mass*ratio)/h
        H = B/ratio
        cx1,cy1 = (N-B)/2,(N+H)/2
        cx2,cy2 = N-cx1,(N+H)/2
        cx3,cy3 = N/2,N-cy1
        draw.polygon([(cx1,cy1),(cx2,cy2),(cx3,cy3)],fill=1)

    elif stype=="pacman":
        draw.chord([(N/4,N*(1.-1./2)/2),(3*N/4,N*(1.+1./2)/2)],
                   30,90,fill=1)
        draw.chord([(N/4,N*(1.-1./2)/2),(3*N/4,N*(1.+1./2)/2)],
                   270,330,fill=1)
        draw.chord([(N/4,N*(1.-1./2)/2),(3*N/4,N*(1.+1./2)/2)],
                   90,270,fill=1)
        draw.polygon([(N/2,N/2), 
                      (N/2+N*np.cos(np.pi/6)/4,N/2+N*np.sin(np.pi/6)/4), 
                      (N/2+N*np.cos(np.pi/2)/4,N/2+N*np.sin(np.pi/2)/4)],
                     fill=1)
        draw.polygon([(N/2,N/2), 
                      (N/2+N*np.cos(-np.pi/6)/4,N/2+N*np.sin(-np.pi/6)/4), 
                      (N/2+N*np.cos(-np.pi/2)/4,N/2+N*np.sin(-np.pi/2)/4)],
                     fill=1)
        draw.ellipse([(N/2,N*0.3),(N/2.+N/10.,N*0.3+N/10)],fill=0)
        adjust_mass = True

    elif stype=="pacman_mod":
        draw.chord([(N/4,N*(1.-2/3)/2),(3*N/4,N*(1.+2/3)/2)],
                   30,90,fill=1)
        draw.chord([(N/4,N*(1.-2/3)/2),(3*N/4,N*(1.+2/3)/2)],
                   270,330,fill=1)
        draw.chord([(N/4,N*(1.-2/3)/2),(3*N/4,N*(1.+2/3)/2)],
                   90,270,fill=1)
        draw.polygon([(N/2,N/2), 
                      (N/2+N*np.cos(np.pi/6)/4,N/2+N*np.sin(np.pi/6)/4), 
                      (N/2+N*np.cos(np.pi/2)/4,N/2+N*np.sin(np.pi/2)/4)],
                     fill=1)
        draw.polygon([(N/2,N/2), 
                      (N/2+N*np.cos(-np.pi/6)/4,N/2+N*np.sin(-np.pi/6)/4), 
                      (N/2+N*np.cos(-np.pi/2)/4,N/2+N*np.sin(-np.pi/2)/4)],
                     fill=1)
        draw.ellipse([(N/2,N*0.3),(N/2.+N/7.,N*0.3+N/7)],fill=0)
        adjust_mass = True
    
    elif stype == "random":
        np.random.seed(42)
        density = extra['density']
        num_points = int(N*N * density)  
        for _ in range(num_points):
            x, y = np.random.randint(N/8, 7*N/8), np.random.randint(N/8, 7*N/8)
            draw.point((x, y), fill=1)

    else:
        raise Exception("Unknown source type.")

    mu=np.asarray(img)

    if adjust_mass:
        m = mu.sum()*h**2
        scaling = np.sqrt(mass/m)

        img = img.resize((int(N*scaling),int(N*scaling)),
                         resample=Image.Resampling.NEAREST)
        w,h = img.size
        delta = int((w-N)/2)
        img = img.crop((delta,delta,delta+N,delta+N)) 
        mu = np.asarray(img)
    
    try:
        angle = extra['rotate']
        img = img.rotate(angle,resample=Image.Resampling.NEAREST)
        mu = np.asarray(img)
    except:
        pass
    
    if blur_kernel is not None:
        mu=signal.convolve2d(mu, blur_kernel, boundary="wrap", mode="same")

    return mu


#compute the exterior transport of the measure mu
def exterior_transport(mu, eps, L, threshold=1e-5, max_iter=1000, 
                       gibbs_kernel=None,init=None):
    
    (N,N_)=mu.shape
    h = L/N
    mu = mu*h**2
    ones = np.ones_like(mu)
    
    if init is not None:
        vecA = init['A']
        vecB = init['B']
    else:
        vecA = np.ones_like(mu)
        vecB = np.ones_like(mu)
    
    oneminusmu=h**2*np.ones_like(mu)-mu
    
    gibbs_kernel = np.array([[np.exp(-((i1-j1)*h)**2/eps) 
                              for i1 in range(0,N)] 
                             for j1 in range(0,N)])
    
    val = -1e6
    k = 0
    while k < max_iter:
        vecB_bar = oneminusmu*vecB
        vecBp=gibbs_kernel@vecB_bar@gibbs_kernel
        vecA=np.divide(ones,vecBp,out=np.ones_like(mu), where=vecBp!=0)
        
        vecA_bar = mu*vecA
        vecAp = gibbs_kernel@vecA_bar@gibbs_kernel
        vecB = np.divide(np.minimum(ones,vecAp),vecAp,out=np.ones_like(mu), 
                         where=vecAp!=0)
        
        margin = (mu.T@(eps*np.log(vecA))).trace()+\
                        (oneminusmu.T@(eps*np.log(vecB))).trace()
        dual_rem = eps*(
            vecB_bar.T@gibbs_kernel@vecA_bar@gibbs_kernel).trace()
        prev_val, val = val, margin - dual_rem
        if np.abs((prev_val-val)/val) <= threshold:
            break
        k += 1

    mass = mu.sum()
    phi = eps*np.log(vecA)
    psi = eps*np.log(vecB) 
    vecB_bar = oneminusmu*vecB
    vecA_bar = mu*vecA
    vecGamma = (gibbs_kernel@vecA_bar@gibbs_kernel)*vecB_bar
    
    cost_kernel = np.array([[((i1-j1)*h)**2*np.exp(-((i1-j1)*h)**2/eps)
                          for i1 in range(0,N)] 
                         for j1 in range(0,N)])

    transport_cost = (vecB_bar.T@cost_kernel@vecA_bar@gibbs_kernel).trace()\
                    +(vecB_bar.T@gibbs_kernel@vecA_bar@cost_kernel).trace()

    return (transport_cost, vecGamma, phi, psi, vecA, vecB)

# W is the double-well function involved in the perimeter approximation
W = lambda s: 0.5*s**2*((1-s)**2)
Wp = lambda s: s*(s-1)*(2*s-1)

# the mass constraint is enforced on Phi instead of Id
Phi = lambda s: (3-2*s)*s**2/6
NormPhi = lambda s: 6*Phi(s)


def double_well_with_forcing_explicit(U, t, eps, xi):
    
    U_w = U*(1-U)*(1-2*U)
    U_s = U - t*(U_w/eps+xi)

    return U_s

def compute_allen_cahn_energy(U,eps,h):
    
    U0 = (U-np.roll(U,1,axis=0))
    U1 = (U-np.roll(U,1,axis=1))
    grad = U0**2+U1**2
    
    return eps*np.sum(grad)/2 + np.sum(W(U))*h**2/eps

def compute_2D_discrete_diffusion_factor(N,delta_x,delta_y,time_step):
    
    freqx = np.fft.fftfreq(N, delta_x)
    freqx = np.fft.fftshift(freqx)
    kx = freqx
    kx2d = np.repeat(kx, N)
    kx2d.shape = (N,N)

    freqy = np.fft.fftfreq(N, delta_y)
    freqy = np.fft.fftshift(freqy)
    ky = freqy
    ky2d = np.repeat(ky, N)
    ky2d.shape = (N,N)
    ky2d = ky2d.T

    symbol = -4*(np.sin(np.pi*kx2d*delta_x)**2/delta_x**2
                 +np.sin(np.pi*ky2d*delta_y)**2/delta_y**2)
    
    return np.exp(symbol*time_step)

def apply_factor_fft(U,factor):
    
    U_f = np.fft.fft2(U)
    U_f = np.fft.fftshift(U_f)
    U_f = U_f*factor
    U_f = np.fft.ifftshift(U_f)
    U_s = np.fft.ifft2(U_f)
    
    return U_s.real

def compute_total_energy(U, eps, h, c, eps_transport, L, usePhi=True):

    if usePhi:
        energy_ups, U_ext, phi, psi, A, B= exterior_transport(NormPhi(U), eps_transport, L)
    else:
        energy_ups, U_ext, phi, psi, A, B= exterior_transport(U, eps_transport, L)
    
    return 6*compute_allen_cahn_energy(U,eps,h) + c*energy_ups


def get_AC_Wasserstein_explicit(U_0, m, t, eps, eps_transport, L, c_W, usePhi=True):
    
    N = U_0.shape[0]
    h = L/N
    U = U_0
    diffusion_factor = compute_2D_discrete_diffusion_factor(N, h, h, eps*t)
    
    def step_func(U, init=None):
        
        U = apply_factor_fft(U, diffusion_factor)
        U[U < 0] = 0
        U[U > 1] = 1
        
        if usePhi:
            energy_ups,U_ext, phi, psi, A, B = exterior_transport(NormPhi(U), eps_transport, L,init=init)
        else:
            energy_ups,U_ext, phi, psi, A, B = exterior_transport(U, eps_transport, L,init=init)
        
        if usePhi:
            U = double_well_with_forcing_explicit(U, t, eps, c_W*(phi-psi)*U*(1-U))
            def optim_func(lmb):
                return np.sum(NormPhi(U+lmb*U*(1-U)))*h**2-m
            lmb,nb_iter = secante(optim_func,0,0.1,100,1e-8)
            U = U+lmb*U*(1-U)

        else:
            U = double_well_with_forcing_explicit(U, t, eps, c_W*(phi-psi)/6)
            def optim_func(lmb):
                return np.sum(U+lmb)*h**2-m
            lmb,nb_iter = secante(optim_func,0,0.1,100,1e-8)
            U = U+lmb
        
        total_energy = 6*compute_allen_cahn_energy(U,eps,h)+c_W*energy_ups

        return U,total_energy,{'A' : A, 'B': B}
    
    
    return step_func


def compute_total_energy(U, eps, h, c, eps_transport, L, usePhi=True):

    if usePhi:
        energy_ups, U_ext, phi, psi, A, B= exterior_transport(NormPhi(U), eps_transport, L)
    else:
        energy_ups, U_ext, phi, psi, A, B= exterior_transport(U, eps_transport, L)
    
    return 6*compute_allen_cahn_energy(U,eps,h) + c*energy_ups


def launch_simulation(m, N, L, h, eps,
                      eps_transport, time_step, nb_iter, c, 
                      shape='disk', extra=None, verbose=True, timelapse=False):
    
    U_0 = build_source(shape, N, L, m, extra=extra)
    energy_init = compute_total_energy(U_0, eps, h, c, 
                                       eps_transport, L)

    U_1 = U_0
    list_of_E, list_of_U = [energy_init], [U_0]

    step_function_1 = get_AC_Wasserstein_explicit(U_0, m, time_step, eps, 
                                                  eps_transport, L, c_W=c)

    nb_prints = 5
    mod = nb_iter/nb_prints
    start = time.time()
    init = None
    i = 0
    prev_E_1, E_1 = 0, -1
    while i < nb_iter and (not np.isclose(E_1, prev_E_1,rtol=1e-8)):
        prev_E1 = E_1
        U_1, E_1, extra = step_function_1(U_1,init)
        init = extra
        list_of_E.append(E_1)
        if i % mod == 0 and i!=0:
            list_of_U.append(U_1)
            if verbose:
                print("i=",i, " E=", E_1)
        i += 1

    if verbose:
        print("i=",i, " prev_E = ", prev_E_1," E=", E_1)
        end = time.time()
        print("Time elapsed ", end - start)

    dic_param ={'shape': shape, 'c': c, 'N': N, 'nb_iter': nb_iter, 
                'm': m, 'L': L,'h': h, 'eps': eps, 
                'eps_transport': eps_transport, 'time_step': time_step}
     
    if timelapse:
        return U_0, U_1, list_of_E, dic_param, list_of_U
    else:
        return U_0, U_1, list_of_E, dic_param, None


def display_figure(U_0, U_1, list_of_E, dic_param, cmap):

    fig  = plt.figure(num=1, clear=True, figsize=(11,9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
    ax_nrg  = fig.add_subplot(gs[0, 0])
    ax_param = fig.add_subplot(gs[0, 1])
    ax_init = fig.add_subplot(gs[1, 0])
    ax_final = fig.add_subplot(gs[1, 1]) 


    data = [(key, round(value, 3) if \
       isinstance(value, float) else value) for key, value in dic_param.items()]
    table = ax_param.table(cellText=data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax_param.axis("off")
    ax_param.set_title('Value of the parameters of the experiment')

    list_of_E = list_of_E[1:]
    N1 = range(len(list_of_E))
    lim = list_of_E[-1]

    im1 = ax_nrg.plot(N1, list_of_E, 'r', label='Energy')
    im2 = ax_nrg.plot(N1, [lim for k in N1], 'b', 
                      label=str("{:.2f}".format(lim)))
    ax_nrg.legend(loc='upper right')
    ax_nrg.set_title("Energy as a function of the number of iterations")
    ax_nrg.grid(linestyle='--')

    im01 = ax_init.imshow(U_0, cmap=cmap)
    im01.set_clim(0,1)
    ax_init.set_title("Initial data")
    cbar_ax_init = fig.add_axes([ax_init.get_position().x1 + 0.015, 
                                 ax_init.get_position().y0, 0.015, 
                                 0.9*ax_init.get_position().height])
    cbar1 = fig.colorbar(im01, cax=cbar_ax_init)

    im10 = ax_final.imshow(U_1, cmap=cmap)
    im10.set_clim(0,1)
    ax_final.set_title(" Final data")
    cbar_ax_final = fig.add_axes([ax_final.get_position().x1 + 0.015, 
                                  ax_final.get_position().y0, 0.015,
                                 0.9*ax_final.get_position().height])
    cbar2 = fig.colorbar(im10, cax=cbar_ax_final)

    plt.savefig('test_display_figure')
    plt.show()
    plt.close()
    
    return None

def display_timelapse(U_0, U_1, list_of_E, dic_param, cmap, list_of_U):


    fig  = plt.figure(clear=True)
    num_col = 5
    gs = fig.add_gridspec(1, num_col)
    
    for idx, U in enumerate(list_of_U):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(U, cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')

        if idx==0:
            x0, y0 = ax.get_position().x0, ax.get_position().y0
            height = ax.get_position().height
            cbar_height = 0.7*height
            center = y0 + height/2
            bottom = center - cbar_height/2
            ax_cbar= fig.add_axes([x0 - 0.01, bottom, 0.01, cbar_height])
            cbar = fig.colorbar(im, cax=ax_cbar)

            #ticks
            cbar.set_ticks([0,1])
            cbar.set_ticklabels([0,1])
            cbar.ax.tick_params(size=0, labelright=False, labelleft=True)


    plt.savefig('test_display_timelapse')
    plt.show()
    plt.close()

def make_gif(list_of_U, cmap):

    frames = []
    for U in list_of_U:
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(U, cmap=cmap)
        ax.axis('off')
    
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    imageio.mimsave('output.gif', frames, duration=2, loop=0)

    return None

