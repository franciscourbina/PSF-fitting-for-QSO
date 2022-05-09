import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import dblquad
import astropy.io.fits as F

# Non-simmetrical Moffat distribution definition.
def moffat(X, A, x_off, y_off,alpha_x, alpha_y, beta, off_set):
    x, y = X
    r_x = ((x-x_off)**2 )/ (alpha_x**2)
    r_y = ((y-y_off)**2 )/ (alpha_y**2)
    return A*(1 + r_x + r_y)**(-beta) + off_set

# 2d Gaussian distribution
def dim2_gauss(X, mu1,sig1, mu2,sig2, A, off_set):
    x,y = X
    res = A *  np.exp(-(x-mu1)**2 / (2*sig1**2)) * np.exp(-(y-mu2)**2 / (2*sig2**2)) + off_set
    return res

# To account for the "fotton counting" process MUSE does, an integration has to be done
# in each pixel, considering its spatial resolution.

# 2d trapezoidal rule
def d2_trapz(function, a, b, c, d, n=10):
    # 2D trapezoidal rule for integrating a function from a to b in x, and from c to d in y.
    # This assumes a uniform grid.
    x_arr = np.linspace(a, b, n)
    y_arr = np.linspace(c, d, n)
    
    X,Y = np.meshgrid(x_arr, y_arr)
    
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]
    Z = function(X,Y)
    
    integration = np.trapz(np.trapz(Z, dx=dy, axis=1), dx=dx, axis=0)
    return integration

def moffat_integrated_func(X, A, x_off, y_off,alpha_x, alpha_y, beta, off_set, delta_x=0.2, delta_y=0.2, 
                    abs_tol=1e-2, rel_tol=1e-2, method = 'scipy',  Moffat=True, n = 10, Ns=30):

    """
    Moffat distribution numerical integration is done over a grid, where each coordinate
    has the center of each pixel of spatial resolution delta_x*delta_y in arcseconds.
    Currently two implementations of this process is avaliable. The first uses Scipy implementation
    of 2d numerical implementation. The second uses a simple trapezoidal approach.
    ----------------------------------------
    Inputs:
    X[np.ndarray] = 2-touple of N*N matrices. Each touple has the x and y coordinates of the grid in flat arrays.
    A, x_off, y_off, alpha_x, alpha_y, beta, off_set [float] = Moffat distribution parameters.
    delta_x, delta_y [float] = spatial resolution of each pixel in arcseconds.
    abs_tol, rel_tol [float] = absolute and relative tolerances, only used in the Scipy implementation.
    method [string] = the integration method to be used. It can be 'scipy' or 'trapz'.
    n [int] = only used when the method is 'trapz'. It is the number of samples for the trapezoidal method.
    Ns [int] = 2d matrix dimension. This is needed to do the conversion from the flat array into 2d matrix.

    Outputs:
    result[np.ndarray] = Ns*Ns length flat array with the Moffat distribution integrated results.
    """
    x,y = X
    
    x = x.reshape(Ns,Ns)
    y = y.reshape(Ns,Ns)
    
    N = np.shape(x)[0]
    M = np.shape(x)[1]
    
    Z =  np.zeros((N,M))
    if Moffat:
        f = lambda x,y: moffat((x,y),A, x_off, y_off,alpha_x, alpha_y, beta, off_set) 
        
    for i in range(N):
        for j in range(M):
            # Defining the integration domain.
            x_lower = x[i,j] - delta_x / 2
            x_upper = x[i,j] + delta_x / 2
            y_lower = y[i,j] - delta_y / 2
            y_upper = y[i,j] + delta_y / 2
            
            if method == 'scipy':
                Z[i,j] = dblquad(f , x_lower, x_upper, y_lower, y_upper, epsabs=abs_tol, epsrel=rel_tol)[0]
            else: 
                Z[i,j] = d2_trapz(f, x_lower, x_upper, y_lower, y_upper, n = n)

    result = Z.ravel()
    return result

def fitting(domain, data, model):
    # It only calls the fitting from curve_fit.
    popt, pcov = curve_fit(model, domain, data)
    return popt