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

def rotated_moffat(X, A, x_off, y_off,alpha_x, alpha_y, beta, off_set, theta=0):
    x, y = X
    x_cord = x.copy()
    y_cord = y.copy()
    # Rotation matrix 
    c, s  = np.cos(theta), np.sin(theta)
    x_cord -= x_off
    y_cord -= y_off
    x_r = c*x_cord - s*y_cord 
    y_r = s*x_cord + c*y_cord
    # Elliptical moffat
    r_x = ((x_r)**2 )/ (alpha_x**2)
    r_y = ((y_r)**2 )/ (alpha_y**2)
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

# Some calculations added.
def FWHM_moffat(alpha_x, alpha_y, beta):
    r_d = (np.abs(alpha_x) + np.abs(alpha_y))/2
    return 2*np.sqrt(2**(1/beta)-1)*r_d

def FWHM_gauss(sigma):
    return 2*np.sqrt(2*np.log(2))*sigma

# Flux scaling function.
def flux_scaling(psf, x_center, y_center, psfcen, winds, wave_slice, data, r_psf_sub, r_psf_scale, delta_lamb=1.25):
    """
    It applies the flux scaling method to substract the PSF from the object.
    ------------------------------
    Inputs:
    psf [np.ndarray] = PSF of the QSO, it can be an empirical or fitted PSF.
    x_center, y_center [float] = x and y coordinates of the center of the object in pixels.
    psf_cen[tuple] = a tuple of the center of the PSF in pixel coordinates.
    winds[np.ndarray] = an array of wavelenght to perform the substraction.
    wave_slice[int] = 
    """
    initial_img = np.zeros((2*r_psf_sub,2*r_psf_sub))
    for w1 in winds:
        w2 = w1 + wave_slice

        #slice the cube between indcies w1:w2, and only focus on region intedended for PSF subtraction
        img = data[w1:w2, (x_center-r_psf_sub):(x_center+r_psf_sub),
            (y_center-r_psf_sub):(y_center+r_psf_sub)].sum(axis=0)*delta_lamb

        #Do same thing, but only for where the PSF is going to be scaled (smaller tha img)
        scale_img = data[w1:w2, (x_center-r_psf_scale):(x_center+r_psf_scale), 
            (y_center-r_psf_scale):(y_center+r_psf_scale)].sum(axis=0)*delta_lamb

        #Create the modelled PSF in the same coordiantes as scale_img
        scale_psf = psf[(psfcen[0]-r_psf_scale):(psfcen[0]+r_psf_scale), 
                        (psfcen[1]-r_psf_scale):(psfcen[1]+r_psf_scale)]

        #Find the scaling necesary to match the modelled PSF to scale_imge
        norm = np.sum(scale_img) / np.sum(scale_psf)

        #Subtract the scaled PSF from the PSF subtraction image (img)
        psf_sub_img = img - norm * psf

        #Mask out the region where the PSF scaling was done with Nans
        psf_sub_img_nan = psf_sub_img.copy()
        psf_sub_img_nan[(psfcen[0]-r_psf_scale):(psfcen[0]+r_psf_scale), \
                    (psfcen[1]-r_psf_scale):(psfcen[1]+r_psf_scale)] = float('nan')

        #Add the scaled image to a borad-banned PSF-subtracted image
        initial_img += psf_sub_img
    
    return initial_img, img, norm