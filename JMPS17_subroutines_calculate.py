import matplotlib.pyplot as plt

from JMPS17_subroutines import *

warnings.simplefilter('ignore')


def det_growth(g1, g2, g3, beta, kh):
    """ Return determinant of 4x4 coefficient matrix for growth and prestretch cases

    Parameters
    ----------
    g1, g2, g3 : float
        values of growth parameters
    beta : float
        stiffness ratio (film/substrate) 
    kh : float
        normalized wavenumber

    Returns
    -------
    dd : float
        determinant of coefficient matrix. 

    Notes
    -----
    If the determinant is too large to compute, returns None.
    """

    LAMBDA = 1. / g1**2. + g1**2. * g3**2.
    AA = numpy.zeros((4, 4), dtype='float64')

    # Eq. 38 in paper
    try:
        AA[0][0] = 2. * (beta * g1**2. * g3**2. + 1.)
        AA[0][1] = 2. * (beta * g1**2. * g3**2. - 1.)
        AA[0][2] = beta * LAMBDA + 2. / g1**2. / g3
        AA[0][3] = beta * LAMBDA - 2. / g1**2. / g3
        AA[1][0] = 2. + beta * LAMBDA
        AA[1][1] = 2. - beta * LAMBDA
        AA[1][2] = 2. * (1. + beta * g3)
        AA[1][3] = 2. * (1. - beta * g3)
        AA[2][0] = 2. * g1**2. * g3**2. * exp(-kh * g1 * g2 * g3)
        AA[2][1] = 2. * g1**2. * g3**2. * exp(kh * g1 * g2 * g3)
        AA[2][2] = LAMBDA * exp(-kh * g2 / g1)
        AA[2][3] = LAMBDA * exp(kh * g2 / g1)
        AA[3][0] = -LAMBDA * exp(-kh * g1 * g2 * g3)
        AA[3][1] = LAMBDA * exp(kh * g1 * g2 * g3)
        AA[3][2] = -2. * g3 * exp(-kh * g2 / g1)
        AA[3][3] = 2. * g3 * exp(kh * g2 / g1)

        dd = numpy.linalg.det(AA)

        if isinf(dd):
            dd = None
            # print("infinity at lam1 = ", 1./g1)

    except (OverflowError):
        dd = None
        # print("overflow at lam1 = ", 1./g1)

    return dd


def det_compression(d1, d3, beta, kh):
    """ Return determinant of 6x6 coefficient matrix for whole-domain compression cases

    Parameters
    ----------
    d1, d3 : float
        values of compression in 1 and 3 directions
    beta : float
        stiffness ratio (film/substrate) 
    kh : float
        normalized wavenumber

    Returns
    -------
    dd : float
        determinant of coefficient matrix. 

    Notes
    -----
    If the determinant is too large to compute, returns None.
    """

    LAMBDA = d1 ** 2. + 1. / (d1 ** 2. * d3 ** 2.)
    AA = numpy.zeros((6, 6), dtype='float64')

    # Eq. 36 in the paper
    try:
        AA[0][0] = 2. * beta
        AA[0][1] = 2. * beta
        AA[0][2] = beta * LAMBDA * d1 ** 2. * d3 ** 2.
        AA[0][3] = beta * LAMBDA * d1 ** 2. * d3 ** 2.
        AA[0][4] = -2.
        AA[0][5] = -LAMBDA * d1 ** 2. * d3 ** 2.

        AA[1][0] = beta * LAMBDA * d3
        AA[1][1] = -beta * LAMBDA * d3
        AA[1][2] = 2. * beta
        AA[1][3] = -2. * beta
        AA[1][4] = LAMBDA * d3
        AA[1][5] = 2.

        AA[2][0] = 2. * exp(-kh / (d1 ** 2. * d3))
        AA[2][1] = 2. * exp(kh / (d1 ** 2. * d3))
        AA[2][2] = LAMBDA * d1 ** 2. * d3 ** 2. * exp(-kh)
        AA[2][3] = LAMBDA * d1 ** 2. * d3 ** 2. * exp(kh)
        AA[2][4] = 0.
        AA[2][5] = 0.

        AA[3][0] = -LAMBDA * d3 * exp(-kh / (d1 ** 2. * d3))
        AA[3][1] = LAMBDA * d3 * exp(kh / (d1 ** 2. * d3))
        AA[3][2] = -2. * exp(-kh)
        AA[3][3] = 2. * exp(kh)
        AA[3][4] = 0.
        AA[3][5] = 0.

        AA[4][0] = -1.
        AA[4][1] = 1.
        AA[4][2] = -d1 ** 2. * d3
        AA[4][3] = d1 ** 2. * d3
        AA[4][4] = -1.
        AA[4][5] = -d1 ** 2. * d3

        AA[5][0] = -1.
        AA[5][1] = -1.
        AA[5][2] = -1.
        AA[5][3] = -1.
        AA[5][4] = 1.
        AA[5][5] = 1.

        dd = numpy.linalg.det(AA)

        if isinf(dd):
            dd = None
            # print( "infinity at d1 = ", d1)

    except (OverflowError):
        dd = None
        # print( "overflow at d1 = ", d1)

    return dd


def eval_FvK(lam1, beta, kh, FvK_options):
    """ Calculate critical axial strain that results in instability according to Foppl von Karman equations

    Parameters
    ----------
    lam1 : float
        values of compression in 1 and 3 directions
    beta : float
        stiffness ratio (film/substrate) 
    kh : float
        normalized wavenumber
    FvK_options : list
        information (mode_type and modifications) for FvK equations

    Returns
    -------
    dd : float
        value of modified critical condition (Eqs. 42 and 44)
    """

    source = FvK_options.split()[0]
    modifications = FvK_options.split()[1]

    if source == 'FvK_growth':
        d1 = 1.
        g1 = 1. / lam1
    elif source == 'FvK_compression':
        d1 = lam1
        g1 = 1.
    else:
        print("there is a problem with the specified mode type")

    if modifications == 'original':
        # original Foppl-von Karman equations
        term = 1.
    elif modifications == 'mod_thickness':
        # Foppl-von Karman equations, modified to use thickness in the deformed configuration according to Eq. 43
        term = g1 / d1
    elif modifications == 'mod_wavelength':
        # Foppl-von Karman equations, modified to use wavelength in the deformed configuration according to Eq. 43
        term = 1. / d1
    elif modifications == 'mod_both':
        # Foppl-von Karman equations, modified to use dimensions in the deformed configuration according to Eq. 43
        term = g1 / d1 ** 2.
    else:
        print("there is a problem with the specified modifications")

    # Eqs. 42 or 44
    dd = ((kh * term) ** 2.) / 3. + 2. / (beta * kh * term) + d1 ** 2. / g1 ** 3. - g1 / d1 ** 2.

    return dd


def Ridder_growth(a, b, g1_function, g2_function, g3_function, beta, kh, tol=1.e-12, nmax=50):
    """ Uses Ridders' method to find critical strain (between a and b) for given wavelength kh

    Parameters
    ----------
    a,b : float
        upper and lower brackets of lambda for Ridders' method
    g1_function, g2_function, g3_function : functions
        functions, determined by set_mode_info, that return values of g for given loading parameter lambda
    beta : float
        stiffness ratio (film/substrate)
    kh : float
        normalized wavenumber
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance
    nmax : int
        maximum number of iterations before exiting

    Returns
    -------
    lambda_crit : float
        value of axial compression, lambda, that satisfies Eq. 38
    n_iter : int
        number of iterations before lambda_crit was found

    Notes
    -----
    Based on based on https://en.wikipedia.org/wiki/Ridders%27_method
    """

    # calculate value at lambda = a
    g1a = g1_function(a)
    g2a = g2_function(a)
    g3a = g3_function(a)
    fa = det_growth(g1a, g2a, g3a, beta, kh)
    if fa == 0.0:
        print("lower bracket is root")
        return a, 0

    # calculate value at lambda = b
    g1b = g1_function(b)
    g2b = g2_function(b)
    g3b = g3_function(b)
    fb = det_growth(g1b, g2b, g3b, beta, kh)
    if fb == 0.0:
        print("upper bracket is root")
        return b, 0

    if fa * fb > 0.0:
        # print("Root is not bracketed between a = {a} and b = {b}".format(a=a, b=b))
        return None, None

    # iterate to find lambda_crit
    for i in range(nmax):
        c = 0.5 * (a + b)
        g1c = g1_function(c)
        g2c = g2_function(c)
        g3c = g3_function(c)
        fc = det_growth(g1c, g2c, g3c, beta, kh)

        s = sqrt(fc ** 2. - fa * fb)
        if s == 0.0:
            return None, i

        dx = (c - a) * fc / s
        if (fa - fb) < 0.0: dx = -dx
        x = c + dx

        g1x = g1_function(x)
        g2x = g2_function(x)
        g3x = g3_function(x)
        fx = det_growth(g1x, g2x, g3x, beta, kh)

        # check for convergence
        if i > 0:
            if abs(x - xOld) < tol * max(abs(x), 1.0):
                return x, i
        xOld = x

        # rebracket root
        if fc * fx > 0.0:
            if fa * fx < 0.0:
                b = x; fb = fx
            else:
                a = x; fa = fx
        else:
            a = c;
            b = x;
            fa = fc;
            fb = fx

    print("Too many iterations")
    return None, nmax


def Ridder_compression(a, b, d1_function, d3_function, beta, kh, tol=1.e-12, nmax=50):
    """ Uses Ridders' method to find critical strain (between a and b) for given wavelength kh

    Parameters
    ----------
    a,b : float
        upper and lower brackets of lambda for Ridders' method
    d1_function, d3_function : functions
        functions, determined by set_mode_info, that return values of d for given loading parameter lambda
    beta : float
        stiffness ratio (film/substrate)
    kh : float
        normalized wavenumber
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance
    nmax : int
        maximum number of iterations before exiting

    Returns
    -------
    lambda_crit : float
        value of axial compression, lambda, that satisfies Eq. 38
    n_iter : int
        number of iterations before lambda_crit was found

    Notes
    -----
    Based on based on https://en.wikipedia.org/wiki/Ridders%27_method
    """

    # calculate value at lambda = a
    d1a = d1_function(a)
    d3a = d3_function(a)
    fa = det_compression(d1a, d3a, beta, kh)
    if fa == 0.0:
        print("lower bracket is root")
        return a, 0

    # calculate value at lambda = b
    d1b = d1_function(b)
    d3b = d3_function(b)
    fb = det_compression(d1b, d3b, beta, kh)
    if fb == 0.0:
        print("upper bracket is root")
        return b, 0

    if fa * fb > 0.0:
        # print("Root is not bracketed between a = {a} and b = {b}".format(a=a, b=b))
        return None, None

    # iterate to find lambda_crit
    for i in range(nmax):
        c = 0.5 * (a + b)
        d1c = d1_function(c)
        d3c = d3_function(c)
        fc = det_compression(d1c, d3c, beta, kh)

        s = sqrt(fc ** 2. - fa * fb)
        if s == 0.0:
            return None, i

        dx = (c - a) * fc / s
        if (fa - fb) < 0.0: dx = -dx
        x = c + dx

        d1x = d1_function(x)
        d3x = d3_function(x)
        fx = det_compression(d1x, d3x, beta, kh)

        # check for convergence
        if i > 0:
            if abs(x - xOld) < tol * max(abs(x), 1.0):
                return x, i
        xOld = x

        # rebracket root
        if fc * fx > 0.0:
            if fa * fx < 0.0:
                b = x; fb = fx
            else:
                a = x; fa = fx
        else:
            a = c;
            b = x;
            fa = fc;
            fb = fx

    res = abs(x - xOld) / max(abs(x), 1.0)

    print("Too many iterations, res = {res}".format(res=res))
    return None, nmax


def Ridder_FvK(a, b, FvK_options, beta, kh, tol=1.e-12, nmax=50):
    """ Uses Ridders' method to find critical strain (between a and b) for given wavelength kh

    Parameters
    ----------
    a, b : float
        upper and lower brackets of lambda for Ridders' method
    FvK_options : list
        list of mode_type (either 'growth' or 'compression') modifications] for Ridder
    beta : float
        stiffness ratio (film/substrate)
    kh : float
        normalized wavenumber
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance
    nmax : int
        maximum number of iterations before exiting

    Returns
    -------
    lambda_crit : float
        value of axial compression, lambda, that satisfies Eq. 38
    n_iter : int
        number of iterations before lambda_crit was found

    Notes
    -----
    Based on based on https://en.wikipedia.org/wiki/Ridders%27_method

    """

    # calculate values at lambda = a,b
    fa = eval_FvK(a, beta, kh, FvK_options)
    if fa == 0.0:
        print("lower bracket is root")
        return a, 0

    fb = eval_FvK(b, beta, kh, FvK_options)
    if fb == 0.0:
        print("upper bracket is root")
        return b, 0

    if fa * fb > 0.0:
        # print("Root is not bracketed between a = {a} and b = {b}".format(a=a, b=b))
        return None, None

    # iterate to find lambda_crit
    for i in range(nmax):
        c = 0.5 * (a + b)
        fc = eval_FvK(c, beta, kh, FvK_options)

        s = sqrt(fc ** 2. - fa * fb)
        if s == 0.0:
            return None, i

        dx = (c - a) * fc / s
        if (fa - fb) < 0.0: dx = -dx
        x = c + dx

        fx = eval_FvK(x, beta, kh, FvK_options)

        # check for convergence
        if i > 0:
            if abs(x - xOld) < tol * max(abs(x), 1.0):
                return x, i
        xOld = x

        # rebracket root
        if fc * fx > 0.0:
            if fa * fx < 0.0:
                b = x; fb = fx
            else:
                a = x; fa = fx
        else:
            a = c;
            b = x;
            fa = fc;
            fb = fx

    res = abs(x - xOld) / max(abs(x), 1.0)

    print("Too many iterations, res = {res}".format(res=res))
    return None, nmax


def check_roots(mode, lam_min, lam_max, npts, beta, kh, plotroots):
    """ Calls root-checking functions depending on mode_type

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    lam_min, lam_max : floats
        min and max min_strains to consider when checking for existence of roots
    npts : int
        number of strain values to consider when checking for existence of roots
    beta : float
        stiffness ratio (film/substrate) 
    kh : float
        normalized wavenumber
    plotroots : boolean
        plot lines showing positive or negative value at all npts for each wavelength
    
    Returns
    -------
    a : float
        smallest lambda value for which a real determinant was calculated (used as lower bound for Ridder algorithm)
    root_exists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    
    Notes
    -----
    
    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    if mode_type == 'growth':
        [g1_function, g2_function, g3_function] = functions
        [root_exists, a] = check_roots_growth(lam_min, lam_max, npts, g1_function, g2_function, g3_function, beta, kh,
                                              mode, plotroots)
    elif mode_type == 'compression':
        [d1_function, d3_function] = functions
        [root_exists, a] = check_roots_compression(lam_min, lam_max, npts, d1_function, d3_function, beta, kh,
                                                   mode, plotroots)
    elif mode_type == 'FvK':
        [root_exists, a] = check_roots_FvK(lam_min, lam_max, npts, functions, beta, kh, mode, plotroots)

    return a, root_exists


def check_roots_growth(lam_min, lam_max, npts, g1_function, g2_function, g3_function, beta, kh, mode, plotroots):
    """ Calculates the value and/or sign of the determinant at every lambda 

    Parameters
    ----------
    lam_min, lam_max : float
        minimum and maximum values of lambda to check for existence of a root
    npts : int
        number of points between lam_min and lam_max at which to calculate determinant
    g1_function, g2_function, g3_function : functions
        functions, determined by set_mode_info, that return values of g for given loading parameter lambda
    beta : float
        stiffness ratio (film/substrate)
    kh : float
        normalized wavenumber
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    plotroots : boolean
        plot lines showing positive or negative value at all npts

    Returns
    -------
    root_exists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    lams[initial] : float
        smallest lambda value for which a real determinant was calculated
    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    lams = numpy.linspace(lam_min, lam_max, npts)  # list of lambda values to calculate determinant for
    dds = numpy.zeros(npts, dtype='float64')  # value of the determinant
    dds_abs = numpy.zeros(npts, dtype='float64')  # sign of the determinant

    root_exists = False
    initial = 0

    for i in range(npts):
        lam1 = lams[i]
        dds[i] = det_growth(g1_function(lam1), g2_function(lam1), g3_function(lam1), beta, kh)

        if isnan(dds[i]):
            dds[i] = 0.0
            dds_abs[i] = 0.0
            initial = i + 1
        else:
            dds_abs[i] = dds[i] / abs(dds[i])
            if dds[i] * dds[initial] < 0.:  # sign change
                root_exists = True

    if plotroots:
        plt.figure()
        plt.axis([0, 1.1, -1.5, 1.5])
        title = "L/H = {wavelength}".format(wavelength=2. * pi / kh)
        plt.title(title)
        plt.xlabel('$\lambda_1$')
        plt.ylabel('energy')

        plt.axvline(x=1., linestyle='--', color='k')
        plt.axhline(y=0., linestyle='--', color='k')
        plt.plot(lams, dds_abs, color=loadcolor, linestyle='-')
        plt.savefig('roots_{mode}_{beta}_{kh:0.3f}.png'.format(mode=mode, beta=beta, kh=kh))

    return root_exists, lams[initial]


def check_roots_compression(lam_min, lam_max, npts, d1_function, d3_function, beta, kh, mode, plotroots):
    """ Calculates the value and/or the sign of the determinant at every lambda 

    Parameters
    ----------
    lam_min, lam_max : float
        minimum and maximum values of lambda to check for existence of a root
    npts : int
        number of points between lam_min and lam_max at which to calculate determinant
    d1_function, d3_function : functions
        functions, determined by set_mode_info, that return values of d for given loading parameter lambda
    beta : float
        stiffness ratio (film/substrate)
    kh : float
        normalized wavenumber
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    plotroots : boolean
        plot lines showing positive or negative value at all npts for each nx

    Returns
    -------
    root_exists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    lams[initial] : float
        smallest lambda value for which a real determinant was calculated
    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    lams = numpy.linspace(lam_min, lam_max, npts)  # list of lambda values to calculate determinant for
    dds = numpy.zeros(npts, dtype='float64')  # value of the determinant
    dds_abs = numpy.zeros(npts, dtype='float64')  # sign of the determinant

    root_exists = False
    initial = 0

    for i in range(npts):
        lam1 = lams[i]
        dds[i] = det_compression(d1_function(lam1), d3_function(lam1), beta, kh)

        if isnan(dds[i]):
            dds[i] = 0.0
            dds_abs[i] = 0.0
            initial = i + 1
        else:
            dds_abs[i] = dds[i] / abs(dds[i])
            if dds[i] * dds[initial] < 0.:  # sign change
                root_exists = True

    if plotroots:
        plt.figure()
        plt.axis([0, 1.1, -1.5, 1.5])
        title = "L/H = {wavelength}".format(wavelength=2. * pi / kh)
        plt.title(title)
        plt.xlabel('$\lambda_1$')
        plt.ylabel('energy')

        plt.axvline(x=1., linestyle='--', color='k')
        plt.axhline(y=0., linestyle='--', color='k')
        plt.plot(lams, dds_abs, color=loadcolor, linestyle='-')
        plt.savefig('roots_{mode}_{beta}_{kh:0.3f}.png'.format(mode=mode, beta=beta, kh=kh))

    return root_exists, lams[initial]


def check_roots_FvK(lam_min, lam_max, npts, FvK_options, beta, kh, mode, plotroots):
    """ Calculates the value and/or the sign of the equation at every lambda 

    Parameters
    ----------
    lam_min, lam_max : float
        minimum and maximum values of lambda to check for existence of a root
    npts : int
        number of points between lam_min and lam_max at which to calculate determinant
    FvK_options : list
        information (mode_type and modifications) for FvK equations 
    beta : float
        stiffness ratio (film/substrate)
    kh : float
        normalized wavenumber
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    plotroots : boolean
        plot lines showing positive or negative value at all npts for each nx

    Returns
    -------
    root_exists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    lams[initial] : float
        smallest lambda value for which a real determinant was calculated
    
    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    lams = numpy.linspace(lam_min, lam_max, npts)  # list of lambda values to calculate determinant for
    dds = numpy.zeros(npts, dtype='float64')  # value of the equation
    dds_abs = numpy.zeros(npts, dtype='float64')  # sign of the equation

    root_exists = False
    initial = 0

    for i in range(npts):
        lam1 = lams[i]
        dds[i] = eval_FvK(lam1, beta, kh, FvK_options)

        if isnan(dds[i]):
            dds[i] = 0.0
            dds_abs[i] = 0.0
            initial = i + 1
        else:
            dds_abs[i] = dds[i] / abs(dds[i])
            if dds[i] * dds[initial] < 0.:  # sign change
                root_exists = True

    if plotroots:
        plt.figure()
        plt.axis([0, 1.1, -1.5, 1.5])
        title = "L/H = {wavelength}".format(wavelength=2. * pi / kh)
        plt.title(title)
        plt.xlabel('$\lambda_1$')
        plt.ylabel('value')

        plt.axvline(x=1., linestyle='--', color='k')
        plt.axhline(y=0., linestyle='--', color='k')
        plt.plot(lams, dds_abs, color=loadcolor, linestyle='-')
        plt.savefig('roots_{mode}_{beta}_{kh:0.3f}.png'.format(mode=mode, beta=beta, kh=kh))

    return root_exists, lams[initial]


def find_roots(root_exists, mode, crit_strains, a, b, c, beta, kh, printoutput, tol=1.e-12):
    """ Calls Ridder algorithm functions depending on mode_type

    Parameters
    ----------
    root_exists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    min_strains : list of floats
        list of critical strain values corresponding to given wavelengths (appended each time this is called)
    a, b, c : floats
        lower, mid, and upper brackets for Ridder's algorithm
    beta : float
        stiffness ratio (film/substrate) 
    kh : float
        normalized wavenumber
    printoutput : boolean
        whether or not to print every root found at every wavelength
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance

    Returns
    -------
    min_strains : list of floats
        list of critical strain values corresponding to given wavelengths (appended each time this is called)
    b : float
        midpoint value of axial compression lambda, used to distinguish between double roots.  Updated each time function is called, and used for the next call

    Notes
    -----
    If root_exists = False, then the strain value of 0 is appended
    :param crit_strains:
    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    minimum = a

    if printoutput:
        print("L/H = {wavelength:0.2f}, a = {a}, b = {b}, c = {c}".format(wavelength=2. * pi / kh, a=a, b=b, c=c))

    if root_exists:
        if mode_type == 'growth':
            [g1_function, g2_function, g3_function] = functions
            [lam1, n] = Ridder_growth(a, c, g1_function, g2_function, g3_function, beta, kh, tol)
        elif mode_type == 'compression':
            [d1_function, d3_function] = functions
            [lam1, n] = Ridder_compression(a, c, d1_function, d3_function, beta, kh, tol)
        elif mode_type == 'FvK':
            [lam1, n] = Ridder_FvK(a, c, functions, beta, kh, tol)

        # in the case of double roots:
        if lam1 is None:
            if mode_type == 'growth':
                [lam1_min, n] = Ridder_growth(a, b, g1_function, g2_function, g3_function, beta, kh, tol)
                [lam1_max, n] = Ridder_growth(b, c, g1_function, g2_function, g3_function, beta, kh, tol)
            elif mode_type == 'compression':
                [lam1_min, n] = Ridder_compression(a, b, d1_function, d3_function, beta, kh, tol)
                [lam1_max, n] = Ridder_compression(b, c, d1_function, d3_function, beta, kh, tol)
            elif mode_type == 'FvK':
                [lam1_min, n] = Ridder_FvK(a, b, functions, beta, kh, tol)
                [lam1_max, n] = Ridder_FvK(b, c, functions, beta, kh, tol)
            # if printoutput: print("double roots: ", lam1_max, lam1_min)
            b = 0.5 * (lam1_min + lam1_max)
            lam1 = lam1_max
        else:
            b = 0.5 * (lam1 + minimum)
        if printoutput: print("lam = {lam:0.5f}, n = {n}".format(lam=lam1, n=n))
    else:
        lam1 = 1.
        n = 1.
        if printoutput: print("L/H = {wavelength:0.2f}, no root".format(wavelength=2. * pi / kh))

    crit_strains.append(1. - lam1)

    return crit_strains, b


def find_critical_values(mode, beta, wavelengths, lam_min, lam_max, npts, plotroots, findroots, printoutput, plotindcurves, tol=1.e-12):
    """ Finds critical strain for each specified wavelength

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    beta : float
        stiffness ratio (film/substrate) 
    wavelengths : list of floats
        list of wavelengths for which to calculate determinant
    lam_min, lam_max : floats
        min and max min_strains to consider when checking for existence of roots
    npts : int
        number of strain values to consider when checking for existence of roots
    plotroots : boolean
        whether or not to plot lines showing positive or negative value at all npts for each wavelength
    findroots : boolean
        whether or not to find the values of each root (set to False and plotroots to True to see root plots)
    printoutput : boolean
        whether or not to print every root found at every wavelength
    plotindcurves : boolean
        whether or not to plot individual energy curves for each beta
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance
    
    Returns
    -------
    strains_masked : list of floats
        list of non-zero min_strains associated with each wavelength
    wavelengths_masks : list of floats
        list of wavelengths associated with non-zero min_strains

    Notes
    -----
    Called by 17JMPS.py
    
    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    a = lam_min  # lower bracket
    c = 0.9999  # upper bracket
    b = 0.5 * (a + c)  # initial middle point

    strains = []

    print("\n {mode} {mode_type} loading, beta = {beta:0.2f}".format(mode=mode, mode_type=mode_type, beta=beta))

    for wavelength in wavelengths:

        kh = 2. * pi / wavelength

        [a, root_exists] = check_roots(mode, lam_min, lam_max, npts, beta, kh, plotroots)

        if findroots:
            [strains, b] = find_roots(root_exists, mode, strains, a, b, c, beta, kh, printoutput, tol)

    if plotindcurves and findroots:
        plt.figure()
        plt.semilogx(wavelengths, strains, color=loadcolor, linestyle='-')
        plt.axis([0.1, 1000., 0., 1.])
        plt.savefig('curve_{mode}_beta={beta:0.1f}.png'.format(mode=mode, beta=beta))

    if findroots:
        # remove zero-strain results
        strains_all = numpy.array(strains)
        strains_masked = numpy.ma.masked_equal(strains, 0)
        zeromask = numpy.ma.getmask(strains_masked)
        wavelengths_masked = numpy.ma.array(wavelengths, mask=zeromask)

        return strains_masked, wavelengths_masked


def write_crit_values(file, values):
    """ Writes all min_strains and min_wavelengths to a file

    Parameters
    ----------
    file : file object
        file object created in JMPS17_calculate.py 
    values : list of floats
        list of critical values (from find_crit_strains)

    Returns
    -------
    None
    
    """

    for value in values:
        if isinstance(value, float):
            try:
                file.write('{value:0.6f} '.format(value=value))
            except TypeError:
                print(value)
    file.write('\n')


def find_min_crit_strain(wavelengths, crit_strains):
    """ Finds minimum critical strain and corresponding min_wavelength

    Parameters
    ----------
    wavelengths : list of floats
        list of wavelengths
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength

    Returns
    -------
    min_wavelength : float
        critical wavelength (corresponding to critical strain)
    min_strain : float
        minimum critical strain
    
    Notes
    -----
    If there is no true minimum, it returns the zero for the wavelength and the zero wavelength strain
    """

    index = 0
    start = True
    for i in range(1, len(wavelengths) - 1):
        if crit_strains[i] == 0.0 and start:
            index = i + 1
        elif crit_strains[i] == 0.0:
            index = index
        elif crit_strains[i] < crit_strains[index]:
            index = i
            start = False

    if abs(crit_strains[-1] - crit_strains[index]) < 0.0001:
        index = len(wavelengths) - 1
        min_wavelength = 0.0
    else:
        min_wavelength = wavelengths[index]

    min_strain = crit_strains[index]

    return min_wavelength, min_strain


def find_threshold_values(beta, mode, crit_strains_all, crit_wavelengths_all, min_strains, min_wavelengths, min_betas, max_strains, max_wavelengths, max_betas, findroots): 
    """ find the threshold (min and max) critical min_strains, and corresponding wavelengths and stiffness ratios

    Parameters
    ----------
    beta : float
        stiffness ratio (film/substrate) 
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    crit_strains_all : list of floats
        list of critical min_strains associated with each wavelength
    crit_wavelengths_all : list of floats
        list of wavelengths
    min_strains : list of floats
        list of minimum strain values
    min_wavelengths : list of floats
        list of wavelengths associated with minimum min_strains
    min_betas : list of floats
        list of stiffness ratios that have a strain minimum associated with them
    max_strains : list of floats
        list of minimum strain values
    max_wavelengths : list of floats
        list of wavelengths associated with maximum min_strains
    max_betas : list of floats
        list of stiffness ratios that have a strain maximum associated with them
    find_roots : boolean
        whether or not to seek values of roots (set to False to only see root plots)

    Returns
    -------
    min_strains : list of floats
        updated list of minimum strain values
    min_wavelengths : list of floats
        updated list of wavelengths associated with minimum min_strains
    min_betas : list of floats
        updated list of stiffness ratios that have a strain minimum associated with them
    max_strains : list of floats
        updated list of minimum strain values
    max_wavelengths : list of floats
        updated list of wavelengths associated with maximum min_strains
    max_betas : list of floats
        updated list of stiffness ratios that have a strain maximum associated with them
    
    Notes
    -----
    what if findroots is False?
    :param mode:
    :param findroots:
    """

    if findroots:

        [min_wavelength, min_strain] = find_min_crit_strain(crit_wavelengths_all, crit_strains_all)
        # for convex curves, find the maximum strain and corresponding wavelength
        [max_wavelength, max_strain] = find_min_crit_strain(crit_wavelengths_all, -crit_strains_all)

        if min_wavelength < 100.:
            min_wavelengths.append(min_wavelength)
            min_strains.append(min_strain)
            min_betas.append(beta)

        if max_wavelength < 30. and max_wavelength != 0.0 and abs(max_wavelength - min_wavelength) > 0.001:
            max_wavelengths.append(max_wavelength)
            max_strains.append(-max_strain)
            max_betas.append(beta)

        return min_strains, min_wavelengths, min_betas, max_strains, max_wavelengths, max_betas


def write_threshold_values(filename, thresh_strains, thresh_wavelengths, betas):
    """ Writes thresh_strains and thresh_wavelengths to a file

    Parameters
    ----------
    filename : string
        filename for saving data
    thresh_strains : list of floats
        list of threshold critical strain for a given stiffness ratio
    thresh_wavelengths : list of floats
        list of thresh_wavelengths associated with each thresh_wavelength
    betas : list of floats
        list of stiffness ratios

    Returns
    -------
    None
    
    """

    with open(filename, 'w') as data_file:
        for i in range(len(thresh_strains)):
            data_file.write('{beta:.4f}\t{strain:.4f}\t{wavelength:.4f}\n'.format(beta=betas[i], strain=thresh_strains[i],
                                                                                  wavelength=thresh_wavelengths[i]))
