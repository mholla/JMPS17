import numpy
from math import *
import warnings

warnings.simplefilter('ignore')


def set_mode_info(mode):
    """ Sets mode_type, loadcolor, and functions for the specified mode

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'

    Returns
    -------
    mode_type: string ('growth', 'compression', or 'buckling')
        determines what function should be used to solve for instability
    loadcolor : string
        hex color for the given mode, to ensure consistent representation across figures
    functions : list
        list of function names that specify amount of growth or compression in terms of axial compression, lambda
    """
    
    # g1 and lam1 are common between all load cases
    g1_function = g_g
    d1_function = d_lam
    

    if mode == 'prestretch1D':
        loadcolor = '#3E8239'
        g2_function = g_inverse
        g3_function = g_one
    elif mode == 'prestretch2D':
        loadcolor = '#2CBF5B'
        g2_function = g_inverse2
        g3_function = g_g
    elif mode == 'unidirectional':
        loadcolor = '#0000CF'
        g2_function = g_one
        g3_function = g_one
    elif mode == 'surface':
        loadcolor = '#6189DC'
        g2_function = g_one
        g3_function = g_g
    elif mode == 'isotropic':
        loadcolor = '#9DBFE8'
        g2_function = g_g
        g3_function = g_g
    elif mode == 'plane':
        loadcolor = '#E02E52'
        d3_function = d_one
    elif mode == 'uniaxial':
        loadcolor = '#EF790F' 
        d3_function = d_inverse
    elif mode == 'biaxial':
        loadcolor = '#DFC223'
        d3_function = d_lam
    elif mode.split()[1] == 'original': 
        loadcolor = '#D3D3D3' 
    elif mode.split()[0] == 'FvK_compression':
        loadcolor = '#A20E0E'
    elif mode.split()[0] == 'FvK_growth':
        loadcolor = '#0C0C57'
    else: 
        loadcolor = 'k'

    if mode in ['prestretch1D', 'prestretch2D', 'unidirectional', 'isotropic', 'surface']:
        mode_type = 'growth'
        functions = [g1_function, g2_function, g3_function] 
    elif mode in ['plane', 'uniaxial', 'biaxial']:
        mode_type = 'compression'
        functions = [d1_function, d3_function]
    else: 
        mode_type = 'FvK'
        functions = mode

    return mode_type, loadcolor, functions

def g_g(lam):
    """ Calculate g = 1/lambda (used in growth and prestretch modes)

    Parameters
    ----------
    lam : float
        value of axial compression lambda

    Returns
    -------
    g : float
        value of g
    """  

    g = 1./lam
    return g

def g_one(lam):
    """ Calculate g = 1 (used in growth and prestretch modes)

    Parameters
    ----------
    lam : float
        value of axial compression lambda

    Returns
    -------
    g : float
        value of g
    """  

    g = 1. 
    return g

def g_inverse(lam):
    """ Calculate g = lambda (used in 1D prestretch mode)

    Parameters
    ----------
    lam : float
        value of axial compression lambda

    Returns
    -------
    g : float
        value of g
    """  

    g = lam
    return g

def g_inverse2(lam):
    """ Calculate g = lambda^2 (used in 2D prestretch mode)

    Parameters
    ----------
    lam : float
        value of axial compression lambda

    Returns
    -------
    g : float
        value of g
    """  
    
    g = lam**2.    
    return g

def d_lam(lam):
    """ Calculate d = lambda (used in biaxial compression mode)

    Parameters
    ----------
    lam : float
        value of axial compression lambda

    Returns
    -------
    d : float
        value of d
    """  

    d = lam
    return d

def d_one(lam):
    """ Calculate d = 1 (used in plane compression mode)

    Parameters
    ----------
    lam : float
        value of axial compression lambda

    Returns
    -------
    d : float
        value of d
    """  

    d = 1.
    return d

def d_inverse(lam):
    """ Calculate d = 1/sqrt(lam) (used in uniaxial compression mode)

    Parameters
    ----------
    lam : float
        value of axial compression lambda

    Returns
    -------
    d : float
        value of d
    """  

    d = 1./sqrt(lam)
    return d

def calc_eff_strain(mode_type, functions, strains):
    """ Calculate effective strain for the given mode_type and min_strains

    Parameters
    ----------
    mode_type: string ('growth', 'compression', or 'buckling')
        determines what function should be used to solve for instability
    functions : list
        list of function names that specify amount of growth or compression in terms of axial compression, lambda
    strains : list of floats
        list of strain values

    Returns
    -------
     : list of floats
        list of effective strain values corresponding to provided min_strains

    """

    if mode_type == 'growth':
        d_3 = d_one
        g_3 = functions[2]
    elif mode_type == 'buckling':
        return strains
    elif mode_type == 'compression': 
        d_3 = functions[1]
        g_3 = g_one

    stretches = 1. - numpy.array(strains)
    strain_eff = numpy.zeros(len(strains))

    for i in range(len(stretches)):
        lam1 = stretches[i]
        lam3 = d_3(lam1)/g_3(lam1)
        
        # Equation found in section 4.3
        strain_eff[i] = 1. - lam1 * sqrt(lam3)

    return strain_eff

def calc_eff_stiffness(mode_type, functions, strains, betas):
    """ Calculate effective stiffness for the given mode_type, min_strains, and stiffness ratios

    Parameters
    ----------
    mode_type: string ('growth', 'compression', or 'buckling')
        determines what function should be used to solve for instability
    functions : list
        list of function names that specify amount of growth or compression in terms of axial compression, lambda
    strains : list of floats
        list of strain values
    betas : list of floats
        list of stiffness ratios (film/substrate) 

    Returns
    -------
    beta_eff : list of floats
        list of effective stiffness values corresponding to provided min_strains and stiffnesses

    """

    if mode_type == 'growth':
        [g_1, g_2, g_3] = functions
    else: # whole-domain compression does not change stiffness ratio
        return betas
    
    stretches = 1. - numpy.array(strains)
    beta_eff = numpy.zeros(len(betas))

    for i in range(len(betas)):
        lam = stretches[i]
        lam1 = 1/g_1(lam)
        lam3 = 1/g_3(lam)

        # Equation found in section 4.3
        beta_eff[i] = betas[i]*(lam1**2. + (lam1*lam3)**(-2.))/2.

    return beta_eff

def calc_eff_wavelength(mode_type, functions, strains, wavelengths):
    """ Calculate effective eff_wavelength for the given mode_type, min_strains, and wavelengths

    Parameters
    ----------
    mode_type: string ('growth', 'compression', or 'buckling')
        determines what function should be used to solve for instability
    functions : list
        list of function names that specify amount of growth or compression in terms of axial compression, lambda
    strains : list of floats
        list of strain values
    wavelengths : list of floats
        list of wavelengths

    Returns
    -------
    wavelength_eff : list of floats
        list of effective wavelength values corresponding to provided min_strains and wavelengths
        
    """

    if mode_type == 'growth':
        [g_1, g_2, g_3] = functions
        [d_1, d_3] = [d_one, d_one]
    else: 
        [g_1, g_2, g_3] = [d_one, d_one, d_one]
        [d_1, d_3] = functions
    
    stretches = 1. - numpy.array(strains)
    wavelength_eff = numpy.zeros(len(wavelengths))

    for i in range(len(wavelengths)):
        lam = stretches[i]
        g1 = g_1(lam)
        g2 = g_2(lam)
        g3 = g_3(lam)
        d1 = d_1(lam)
        d3 = d_3(lam)

        # Equation found in section 4.3
        wavelength_eff[i] = wavelengths[i]*d1*d1*d3/(g1*g2*g3)

    return wavelength_eff

def calc_axial_pressure(mode, strains):
    """ Calculates axial or Biot pressure generated by a given strain

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    strains : list of floats
        list of strain values

    Returns
    -------
    P_biot : list of floats
        list of Biot pressure values corresponding to the given min_strains

    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    n = len(strains)
    stretches = numpy.array(1. - numpy.array(strains))
    P_biot = numpy.zeros(len(strains))

    if mode_type == 'growth':
        g_1 = functions[0]
        g_2 = functions[1]
        g_3 = functions[2]
        d_1 = d_one
        d_3 = d_one
    elif mode_type == 'buckling':
        g_2 = g_one
        g_3 = g_one
        d_3 = d_one
        if mode.split()[0] == 'compression':
            g_1 = g_one
            d_1 = d_lam
        elif mode.split()[0] == 'growth':
            g_1 = g_g
            d_1 = d_one
    elif mode_type == 'compression': 
        g_1 = g_one
        g_2 = g_one
        g_3 = g_one
        d_1 = functions[0]
        d_3 = functions[1]

    for i in range(len(stretches)):
        g1 = g_1(stretches[i])
        g2 = g_2(stretches[i])
        g3 = g_3(stretches[i])
        d1 = d_1(stretches[i])
        d3 = d_3(stretches[i])

        # Eq. 39
        P_biot[i] = -(d1**2. / g1**2. - g1**2. * g3**2. / (d1**2. * d3**2.))/(g1 * g2 * g3)
        
    return P_biot 

def calc_hydro_pressure(mode, strains):
    """ 

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    strains : list of floats
        list of strain values

    Returns
    -------
    P_hydro : list of floats
        list of hydrostatic pressure values corresponding to the given min_strains

    Notes
    -----

    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    n = len(strains)
    stretches = numpy.array(1. - numpy.array(strains))
    P_hydro = numpy.zeros(len(strains))

    if mode_type == 'growth':
        g_1 = functions[0]
        g_2 = functions[1]
        g_3 = functions[2]
        d_1 = d_one
        d_3 = d_one
    elif mode_type == 'buckling':
        g_2 = g_one
        g_3 = g_one
        d_3 = d_one
        if mode.split()[0] == 'compression':
            g_1 = g_one
            d_1 = d_lam
        elif mode.split()[0] == 'growth':
            g_1 = g_g
            d_1 = d_one
    elif mode_type == 'compression': 
        g_1 = g_one
        g_2 = g_one
        g_3 = g_one
        d_1 = functions[0]
        d_3 = functions[1]

    for i in range(len(stretches)):
        g1 = g_1(stretches[i])
        g2 = g_2(stretches[i])
        g3 = g_3(stretches[i])
        d1 = d_1(stretches[i])
        d3 = d_3(stretches[i])

        lam1 = d1/g1
        lam3 = d3/g3
        lam2 = 1./lam1/lam3
        J = g1*g2*g3

        # Eq. 40
        P_hydro[i] = -(lam1**2 + lam3**2. - 2*lam2**2.)/(3.*J)
        
    return P_hydro 
