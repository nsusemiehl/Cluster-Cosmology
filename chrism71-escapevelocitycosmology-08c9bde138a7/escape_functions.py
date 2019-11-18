from __future__ import division
from math import *
import math
from astropy import units as u
from sympy import *
from sympy.matrices import *
import numpy as np
import pylab as p
import scipy
import astropy
import scipy.special as ss
import astropy.constants as astroc
import matplotlib.pyplot as plt
import scipy.integrate as integrate

######## constants ########
Msun = 1.9891e+30 #kg
c = 299792.458 # km/s


"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		     ESCAPE FUNCTIONS	           ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""


#total mass of a halo from Retana-Montenegro et al. 2012; f-la (8) in terms of masses of the Sun. [Msun]
def M_total(rho_0, h, n):
    return 4.*np.pi*rho_0*(h**3.)*n*ss.gamma(3.*n)

#define Einasto potential profile [(km/s)^2] need to multiply by 3.24077929e-29 to keep units correct
def phi_einasto(r,rho_0,h,n):
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass).value #Mpc km2/s^2 Msol
    part1 = np.array((r/h)**(1./n))
    part2 = np.array(r/h)
    part3 = np.array(M_total(rho_0, h, n)/r)
    return -G_newton*part3*(1. - ss.gammaincc(3.*n, part1) +  part2*ss.gamma(2.*n)*ss.gammaincc(2.*n, part1)/ss.gamma(3.*n) )

def phi_nfw(r,rho_s,r_s):
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass).value #Mpc km2/s^2 Msol
    return -4*np.pi*G_newton*rho_s.value*(r_s.value**2.0)*np.log(1+r/r_s.value)/(r/r_s.value)
    

#escape velocity in Einasto GR [km/s]
def v_esc_einasto(theta,z,rho_0,h,n,N,cosmology):
    """
    Returns the line-of-sight escape velocity profile for a cluster with an Einasto density profile and a cosmology.
    
    theta  -> Requires an array of angles on the sky at which the escape velocity is inferred.
    z      -> redshift
    rho_0, h, n  -> The Einasto shape parameters from https://www.aanda.org/articles/aa/full_html/2012/04/aa18543-11/aa18543-11.html
    N -> The number of galaxies in the phase space. 
         This is used to provide a suppressed line-of-sight escape velocity. For an unsuppressed escape edge, use N = 1000000
    cosmology -> Astropy cosmology object. 
                 Requires a "name", e.g., cosmo = wCDM(H0=70, Om0=0.2,Ode0 = 0.8,w0=-1,name = 'wCDM'). See q_z_function.
    """

    H_z = cosmology.H(z).value
    q_z= q_z_function(z,cosmology)
    r = theta * cosmology.angular_diameter_distance(z)
    if q_z < 0.:
        req = r_eq(z,M_total(rho_0, h, n),cosmology)
        v_esc =  (-2.*phi_einasto(r,rho_0,h,n)+2.*phi_einasto(req,rho_0,h,n) - q_z*(H_z**2.)*(r**2. - req**2.)  )**0.5
    elif q_z >= 0.:
        v_esc =  ( -2.*phi_einasto(r,rho_0,h,n))**0.5
                  
    v_esc_projected = v_esc / Zv(N)
    return r,v_esc_projected


def v_esc_NFW_M200(theta,z,M200,N,cosmology):
    """
    Returns the line-of-sight escape velocity profile for a cluster with an NFW density profile and a cosmology.
    In this case, the density profile is completely described by M200 and the mass-concentration relation used in
    the Sereno meta-catalog.
    
    theta  -> Requires an array of angles on the sky at which the escape velocity is inferred.
    z      -> redshift
    M200_1e14 -> The mass at 200 times the critical density normalized to 1e14.
    N -> The number of galaxies in the phase space. 
         This is used to provide a suppressed line-of-sight escape velocity. For an unsuppressed escape edge, use N = 1000000
    cosmology -> Astropy cosmology object. 
                 Requires a "name", e.g., cosmo = wCDM(H0=70, Om0=0.2,Ode0 = 0.8,w0=-1,name = 'wCDM'). See q_z_function.
    """
    H_z = cosmology.H(z).value
    q_z= q_z_function(z,cosmology)
    r = theta * cosmology.angular_diameter_distance(z)
    rho_crit =  cosmology.critical_density(z)
    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    r200 =   (3*M200.value/(4*np.pi*200*rho_crit.value))**(1/3.0)
    Mtot = M200
    c200 =  concentration_meta(M200,z,cosmology)
    g = (np.log(1+c200.value) - (c200.value/(1+c200.value)))**(-1.)
    rho_s = (M200.value/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass).value #Mpc km2/s^2 kg

    """if r_eq -> infinity then return vanilla Einasto"""
    if q_z < 0.:
        req = r_eq(z,Mtot,cosmology)
        v_esc = (-2.*phi_nfw(r,rho_s,r_s) +2*phi_nfw(req.value,rho_s,r_s)-q_z*(H_z**2.)*(r**2 - req.value**2) )**0.5
    elif q_z >= 0.:
        v_esc = np.sqrt(-2.*phi_nfw(r,rho_s,r_s))

    v_esc_projected = v_esc / Zv(N)

    return r, v_esc_projected


"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		   SUPPRESSION        			   ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""

def Zv(N):
    """
    Returns the value of the line-of-sight statistical suppression
    N -> Number of tracers in the phase-space. 
    """
    
    N0 = 14.205
    lam = 0.467
    return 1 +(N0/N)**lam


"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		  HOME GROWN COSMOLOGY 			   ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
def concentration_meta(mass,redshift,cosmology):
    """
    input m200 & cosmology --> c200
    (concentration relation used in Sereno's metacatalog)

    NOTE: input masses must be in same cosmology as little_h listed
    and in units of Msun
    """
    A = 5.71 
    B = -0.084 
    C= -0.47 
    Mpivot = 2e12/cosmology.h

    c200 = A * (mass/Mpivot)**B * (1+redshift)**C
    
    return c200

def q_z_function(z,cosmology):

    """ 
    Returns the deceleration parameter as a function of redshift (z_c) for three cases
    corresponding to different sets of cosmological parameters:
    
    I)     'FlatLambdaCDM' takes in cosmo_params = w, omega_M
    II)    'wCDM' takes in cosmo_params = w0, wa, omega_M
    III)   'LambdaCDM'
    IV)    'FlatwCDM'
    V)     'Flatw0waCDM'
    NOTE: 'wCDM' assumes the CPL parametrization of dark energy

    """
    case = cosmology.name
    if case == 'Flatw0waCDM':
        w0 = cosmology.w0
        wa = cosmology.wa
        omega_M = cosmology.Om0
        
        #assume flatness:
        omega_DE = 1. - omega_M 

        E_z= np.sqrt(omega_M*(1 + z)**3. + (omega_DE*(1 + z)**(3*(1 + w0 + wa))) * np.exp(-(3*wa*z)/(1 + z)) )

        omega_M_z = (omega_M * (1+z)**3.) / E_z**2.
        omega_DE_z = (omega_DE*(1+z)**(3*(1+w0+wa)) *np.exp(-3*wa*z/(1+z)) ) / E_z**2.

        return (omega_M_z + omega_DE_z*(1 + 3*w0 + (3*wa*z/(1+z)) ) )/2.

    elif case == 'FlatwCDM':
        w = cosmology.w0
        omega_M = cosmology.Om0

        #assume flatness:
        omega_DE = 1. - omega_M

        E_z= np.sqrt( omega_DE * (1+z)**(3.+ 3.*w)  + omega_M * (1+z)**3. )
        omega_M_z = (omega_M * (1+z)**3.) / E_z**2.
        omega_DE_z = (omega_DE*(1+z)**(3.+3.*w)) / E_z**2.

        return ( omega_M_z + omega_DE_z*(1. + 3.*w) )/2.
  
    elif case == 'wCDM':
        w = cosmology.w0
        omega_M = cosmology.Om0
        omega_DE = cosmology.Ode0
        E_z= np.sqrt( omega_DE * (1+z)**(3.+ 3.*w)  + omega_M * (1+z)**3. )
        omega_M_z = (omega_M * (1+z)**3.) / E_z**2.
        omega_DE_z = (omega_DE*(1+z)**(3.+3.*w)) / E_z**2.

        return ( omega_M_z + omega_DE_z*(1. + 3.*w) )/2.

    elif case == 'LambdaCDM':
        omega_DE = cosmology.Ode0
        omega_M = cosmology.Om0

        #omega_K != 0
        E_z= np.sqrt( omega_DE  + omega_M * (1+z)**3. + (1. - omega_DE - omega_M ) * (1+z)**2. )

        omega_M_z = ( omega_M * (1+z)**3. ) / E_z**2.
        omega_DE_z = omega_DE / E_z**2.

        return (omega_M_z/2.) - omega_DE_z 
    
    elif case == 'FlatLambdaCDM':
        omega_M = cosmology.Om0
        
        #flat,
        omega_DE = 1.- omega_M  

        E_z= np.sqrt( omega_DE  + omega_M * (1+z)**3.)

        omega_M_z = ( omega_M * (1+z)**3. ) / E_z**2.
        omega_DE_z = omega_DE / E_z**2.

        return (omega_M_z/2.) - omega_DE_z 
    
def z_trans(cosmology):
    """
    Returns redshift at which q_z drops to zero.
    
    cosmology -> An astropy cosmology object.
    """
        
    redshift_array= np.arange(0,1.4,1e-7)         
    q_z_array = q_z_function(redshift_array,cosmology)

    return redshift_array[np.abs(np.subtract.outer(q_z_array, 0.)).argmin(0)]


def r_eq(z,M,cosmology):
    """
    Returns the  equivalence radius in Mpc for all cosmology cases

    """
    H_z = cosmology.H(z)
    G_newton = astroc.G.to( u.Mpc *  u.km**2 / u.s**2 / u.solMass).value #Mpc km2/s^2 Msol
    r_eq_cubed = -((G_newton*M) / (q_z_function(z, cosmology) * H_z.value**2.)) #Mpc ^3
    r_eq = (r_eq_cubed)**(1.0/3.0)# Mpc
    
    return r_eq

"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"" 		   DENSITY PROFILES  	           ""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""


"""define mass perc error"""
#statistical mass error on M200 increases if  cosmology is not fixed, as such, the following uncertainties in M200
# (for M200 = 4e14) are doubled (5% becomes 10% and so on)

###Nominal NFW Density Profile###
def rhos_nfw(r,rho_s,r_s):
    """
    Radial rho NFW 
    INPUT: r is radial in [Mpc]
           rho_s is the scale density Msol/Mpc^3
           r_s is the scale radius (Mpc)
    OUTPUT:  Msun/Mpc^3
    """
    return rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

def rhos_nfw_int(r,rho_s,r_s):
    """
    Radial r^2*rho NFW integrand for Mass measurement
    INPUT: r is radial in [Mpc]
           rho_s is the scale density Msol/Mpc^3
           r_s is the scale radius (Mpc)
    OUTPUT:  Msun/Mpc^3
    """
    return r**2 * rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

###NFW Density Profile given just M200 and an M-c relation (e.g., Sereno)###
def rho_nfw_m200(r,m200,z,cosmology):
    
    """
    Radial rho NFW 
    INPUT: r is radial in [Mpc]
           m200 is the mass inside r200 (Msol)
           z is the redshift
           cosmology is an astropy object
    OUTPUT:  Msun/Mpc^3
    """
    rho_crit = cosmo.critical_density(z)
    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    r200 =   (3*m200/(4*np.pi*200*rho_crit.value))**(1/3.0)
    c200 =  concentration_meta(m200,z,cosmology)
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    rho_s = (m200/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius

    return rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

###NFW Density Profile given M200, R200, and concentration###
def rho_nfw(r,m200,r200,c200):
    """
    Radial rho NFW 
    INPUT: r is radial in [Mpc]
           m200 is the mass inside r200 (Msol)
           r200 is physical radius containing 200 times critical density
           c200 is NFW concentration parameter
    OUTPUT:  Msun/Mpc^3
    """
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    rho_s = (m200/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius

    return rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

def rho_nfw_int(r,m200,r200,c200):
    """
    Radial r^2*rho NFW integrand for Mass measurement
    INPUT: r is radial in [Mpc]
           m200 is the mass inside r200 (Msol)
           r200 is physical radius containing 200 times critical density
           c200 is NFW concentration parameter
    OUTPUT:  Msun/Mpc^3
    """
    g = (np.log(1+c200) - (c200/(1+c200)))**(-1.)
    rho_s = (m200/(4.*np.pi*r200**3.)) * c200**3. * g
    r_s = r200/c200 #scale radius
    return r**2 * rho_s / ( (r/r_s) *  (1+ r/r_s)**2.  )

###Nominal Einasto Density Profile###
def rho_einasto(r,rho_0, h, n):    
    """
    Radial rho Einasto 
    INPUT: r is radial in [Mpc]
           rho_0 is the half density Msol/Mpc^3
           h is the half radius (Mpc)
           n is the outer slope (?)
    OUTPUT:  Msun/Mpc^3
    """
    return rho_0*np.exp(-(r/h)**(1./n))

def rho_einasto_int(r,rho_0, h, n):
    
    """
    Radial r^2*rho Einasto integrand for Mass measurement.
    INPUT: r is radial in [Mpc]
           rho_0 is the half density Msol/Mpc^3
           h is the half radius (Mpc)
           n is the outer slope (?)
    OUTPUT:  Msun/Mpc^3
    """
    return r**2*rho_0*np.exp(-(r/h)**(1./n))

"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
" 		  TOOLS  	                        "
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""

def einasto_nfwM200_errors(M200, mass_perc_error, z,cosmology):
    rho_crit = cosmology.critical_density(z)
    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    R200 =  (3*M200/(4*np.pi*200.* rho_crit))**(1./3.)
    conc =  concentration_meta(M200.value,z,cosmology)
    r_array_fit = np.arange(0.1,3,0.01)
    ein_params_guess= [1e15,0.5,0.5]
    R200_uncertainty = (R200.value / 3.) * (mass_perc_error)
    M200_deltaM200_plus =  M200.value + (M200.value * mass_perc_error)
    c200_deltac200_plus = concentration_meta(M200_deltaM200_plus,z,cosmology)
    R200_deltaR200_plus = R200.value + R200_uncertainty
    end_r200, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200.value),2)+0.005) & (r_array_fit > round(find_nearest(r_array_fit,R200.value),2)-0.005) )[0]
    rho_ein_params_array = scipy.optimize.curve_fit(rho_einasto,r_array_fit[0:end_r200+1], rho_nfw(r_array_fit[0:end_r200+1],M200.value,R200.value,conc),p0=ein_params_guess)
    n =rho_ein_params_array[0][2]
    h= rho_ein_params_array[0][1]
    rho_0 = rho_ein_params_array[0][0]
    end_r200_plus, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200_deltaR200_plus),2)+0.005) & (r_array_fit > round(find_nearest(r_array_fit,R200_deltaR200_plus),2)-0.005) )[0]
    delta_plus_rho_ein_params_array = scipy.optimize.curve_fit(rho_einasto,r_array_fit[0:end_r200_plus+1], rho_nfw(r_array_fit[0:end_r200_plus+1],M200_deltaM200_plus,R200_deltaR200_plus,c200_deltac200_plus),p0=ein_params_guess)
    delta_plus_n =delta_plus_rho_ein_params_array[0][2]
    delta_plus_h= delta_plus_rho_ein_params_array[0][1]
    delta_plus_rho_0 = delta_plus_rho_ein_params_array[0][0]
    sigma_n = np.abs(n-delta_plus_n)
    sigma_h = np.abs(h - delta_plus_h)
    sigma_rho_0 = np.abs(rho_0-delta_plus_rho_0)

    return M200,R200, conc, rho_0, h,n, sigma_rho_0, sigma_h, sigma_n


def nfws_errors(M200, mass_perc_error, z,cosmology):
    rho_crit = cosmology.critical_density(z)
    rho_crit = rho_crit.to(u.solMass / u.Mpc**3)
    R200 =  (3*M200/(4*np.pi*200.* rho_crit))**(1./3.)
    conc =  concentration_meta(M200.value,z,cosmology)
    r_array_fit = np.arange(0.1,3,0.01)
    nfw_params_guess= [1e15,0.3]
    R200_uncertainty = (R200 / 3.) * (mass_perc_error)
    M200_deltaM200_plus =  M200 + (M200 * mass_perc_error)
    c200_deltac200_plus = concentration_meta(M200_deltaM200_plus.value,z,cosmology)
    R200_deltaR200_plus = R200 + R200_uncertainty
    end_r200, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200.value),2)+0.005) & (r_array_fit > round(find_nearest(r_array_fit,R200.value),2)-0.005) )[0]
#   fit2 = curve_fit(lambda x, a, c: parabola(x, a, b_fixed, c), x, y) 

    rho_nfw_params_array = scipy.optimize.curve_fit(lambda r,rho_s,r_s: rhos_nfw(r,rho_s,r_s),
       r_array_fit[0:end_r200+1], rho_nfw(r_array_fit[0:end_r200+1],M200.value,R200.value,conc),p0=nfw_params_guess,maxfev = 100000)
    rho_s_fit =rho_nfw_params_array[0][0]
    r_s_fit = rho_nfw_params_array[0][1]
    end_r200_plus, = np.where( (r_array_fit < round(find_nearest(r_array_fit,R200_deltaR200_plus.value),2)+0.005) & (r_array_fit > round(find_nearest(r_array_fit,R200_deltaR200_plus.value),2)-0.005) )[0]
    delta_plus_rho_nfw_params_array = scipy.optimize.curve_fit(lambda r,rho_s,r_s: rhos_nfw(r,rho_s,r_s), 
        r_array_fit[0:end_r200_plus+1],
        rho_nfw(r_array_fit[0:end_r200_plus+1],M200_deltaM200_plus.value,R200_deltaR200_plus.value,c200_deltac200_plus),
        p0=nfw_params_guess,maxfev = 100000)
    delta_plus_rho_s =delta_plus_rho_nfw_params_array[0][0]
    delta_plus_r_s = delta_plus_rho_nfw_params_array[0][1]
    sigma_rho_s = np.abs(rho_s_fit - delta_plus_rho_s)
    sigma_r_s = np.abs(r_s_fit - delta_plus_r_s)
    return M200,R200, conc, rho_s_fit, sigma_rho_s, r_s_fit, sigma_r_s

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

