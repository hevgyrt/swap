""" Script containing a selection of idealized theoretical spectrums
"""

import numpy as np

def fully_developed_pm(U10,f):
    """ Compute the theoretical fully developed wind sea Pierson-Moskowitz
    spectrum

    Args:
        U10: 10m wind [m/s]
        f: frequency array (numpy array)

    Returns:
        variance density spectrum (in frequency)
    """
    g = 9.81
    f_pm_tilde = 1/7.69
    f_pm = f_pm_tilde*(g/U10)

    #print("f_PM= {} Hz".format(round(f_pm,4)))
    #j PM spectrum for fully developed wind sea
    alpha_pm = 0.0081

    pm_fully_developed = alpha_pm*g**2*(2*np.pi)**(-4)*f**(-5)*np.exp(-1.25*(f/f_pm)**(-4))
    pm_sigma = pm_fully_developed*(1/(2*np.pi))

    return pm_fully_developed

def JONSWAP(f, fp, U10):
    """ Compute the theoretical JONSWAP spectrum

    Args:
        f: frequency array (numpy array)
        fp: peak frequency (scalar)
        U10: 10m wind [m/s]

    Returns:
        variance density spectrum (in frequency)
    """
    g = 9.81
    # dimensionelss f_p
    f_tilde_p = fp*(U10/g)  # Eq.(6.3.5) Holthuijsen (2007)

    alpha = 0.0317*f_tilde_p**(0.67)

    # peak enhancement and spectral width from JONSWAP (p.162)
    gamma = 3.3
    sigma_a = 0.07
    sigma_b = 0.09

    # Transition to fully deveopled sea staet (see Eq. (6.3.18))
    #gamma = 5.870*f_tilde_p**(0.86)
    #sigma_a = 0.0547*f_tilde_p**(0.32)
    #sigma_b=0.0783*f_tilde_p**(0.16)

    #print("Dimensionless f_peak: {}\n".format(round(f_tilde_p,3)))
    #print(" Alpha = {},\n gamma = {},\n sigma_a = {},\n sigma_b = {}".format(alpha,gamma,sigma_a,sigma_b))

    #Compute PM shape:
    pm_shape = alpha*(g**2)*((2*np.pi)**(-4))*(f**(-5))*np.exp(-1.25*(f/fp)**(-4))
    #Compute peak_enhancement_function:
    G_f = gamma**(np.exp(-.5 * ((f/(fp**-1))/sigma_b)))

    jonswap = pm_shape*G_f
    return jonswap

def donelan(f, fp, cp, U10, implemented_type="holt",C_d=None):
    """
    """
    g = 9.81
    omega_p = 2*np.pi*fp
    alpha_toba = 0.096 # See Holthuijsen p 157 Note 6C
    wave_age = U10/cp


    if not C_d:
        C_d = (0.75 + 0.0067*U10)*1e-4

    u_star = np.sqrt(C_d)*U10


    # Alpha Donelan:
    alpha_donelan = alpha_toba*u_star*omega_p/g * 2*np.pi
    beta = 0.006 * wave_age**0.55
    print("alpha: {}, beta: {}".format(alpha_donelan, beta))

    # gamma (peak enhancement factor)
    if wave_age < 1:
        if wave_age < 0.83:
            print("Wave age = {}, i.e. less than 0.83".format(wave_age))
        gamma_don = 1.7
    else:
        if wave_age > 5:
            print("Wave age = {}, i.e. more than 5".format(wave_age))
        gamma_don = 1.7 + 6*np.log(wave_age)

    # spectral width
    sigma_don = 0.08*(1 + 4/(wave_age**3))


    if implemented_type == "holt":
        # Donelan after Holthuijsen
        # Compute Donelan shape:
        don_shape = alpha_donelan*(g**2)*((2*np.pi)**(-4))*(f**(-4))*(fp**(-1))* np.exp(-(f/fp)**(-4))
        #Compute peak_enhancement_function:
        don_G = gamma_don**(np.exp(-.5 * ((f/(fp**-1))/(sigma_don)**2)))
    elif implemented_type == "rogers":
        # Donelan after Rogers (2012)
        don_shape = (beta*(g**2)) / ( ((2*np.pi)**4) * (f**4) * fp ) * np.exp(-(fp/f)**4)
        don_G = gamma_don**(np.exp(-.5 * ((f-fp)/(sigma_don*fp))**2))
    else:
        raise Exception('implemented_type must be either "holt" or "rogers"')


    donelan = don_shape*don_G

    return donelan
