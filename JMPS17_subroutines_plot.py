import matplotlib.pyplot as plt

from JMPS17_subroutines import *

warnings.simplefilter('ignore')


def other_analytical(betas):
    """ Calculates analytical solution from Cai & Fu 1999 (doi 10.1098/rspa.1999.0451)

    Parameters
    ----------
    betas : list of floats
        list of stiffness ratio (film/substrate) 

    Returns
    -------
    None

    """

    # In Cai & Fu 1999, they denote the stiffness ratio (substrate/film) as r
    rs = 1. / betas

    # Eq. 3.25 in Cai & Fu 1999
    stretches = 1. - (3. * rs) ** (2. / 3.) / 4. + 33. * rs / 160. * (3. * rs) ** (1. / 3.)
    khs = (3. * rs) ** (1. / 3.) + 3. * rs / 20.

    lhs = 2. * pi / khs
    strains = 1. - stretches

    plt.figure('10A')
    plt.plot(betas, strains, color='#ffd700', linewidth=3, linestyle='--', zorder=1)
    data_strains('10A', color=False)

    plt.figure('10B')
    plt.plot(betas, lhs, color='#ffd700', linewidth=3, linestyle='--', zorder=1)
    data_wavelength('10B', color=False)


def datapoints():
    """ Call other functions to add datapoints to figures

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    data_wavelength('5B')
    data_strains('5A')
    data_eff_wavelength()
    data_eff_strain()


def data_Tallinen2015(figno=None, plot=False):
    """ Calculate wavelengths and

    Parameters
    ----------
    plot : boolean
        if True, adds Tallinen data points to Figs. 4 and 9

    Returns
    -------
    betas : list of floats
        list of stiffness ratios associated with Tallinen 2015 data
    min_strains : list of floats
        list of min_strains associated with Tallinen 2015 data
    wavelengths : list of floats
        list of wavelengths associated with Tallinen 2015 data

    """

    # calculations for Tallinen et al 2015 (doi 10.1103/PhysRevE.92.022720)
    beta = numpy.array([0.86, 3, 6, 10])
    # Tallinen et al 2015, Fig. 4d (zig zags)
    g1 = numpy.array([1.6680328, 1.3196721, 1.2110655, 1.1495901])
    lh1 = numpy.array([3.1440163, 4.0162272, 5.5172415, 6.572008])
    g2 = numpy.array([1.6680328, 1.3217213, 1.2090164, 1.1495901])
    lh2 = numpy.array([3.1845841, 5.233266, 7.7079105, 9.817444])
    # Tallinen et al 2015, Fig. 4d (triple junctions)
    g3 = numpy.array([1.6721171, 1.3234556, 1.2116568, 1.1548723])
    lh3 = numpy.array([3.3809805, 4.9423223, 7.1327753, 8.89265])

    betas = numpy.concatenate((beta, beta, beta))
    gs = numpy.concatenate((g1, g2, g3))
    strains = 1. - 1. / gs
    wavelengths = numpy.concatenate((lh1, lh2, lh3))

    # add data to Figs. 4 and 9
    if plot:
        plt.figure('{figno}_{subfigno}_{mode}'.format(figno=figno, subfigno=4, mode='prestretch2D'))
        plt.semilogx(
            wavelengths,
            strains,
            linestyle='',
            color=set_mode_info('prestretch2D')[1],
            markeredgecolor='k',
            marker='*',
            markersize=10
        )

    return betas, strains, wavelengths


def data_strains(figname, color=True):
    """ add strain datapoints to Fig 5A

    Parameters
    ----------
    figname : string
        name of figure for datapoints to be added to
    color : boolean (default = True)
        determines if datapoints should be added in their specific loadcolor, or all gray

    Returns
    -------
    None

    """

    plt.figure(figname)

    if color:
        color_plane = set_mode_info('plane')[1]
        color_uniaxial = set_mode_info('uniaxial')[1]
        color_biaxial = set_mode_info('biaxial')[1]
        color_surface = set_mode_info('surface')[1]
        color_prestretch1D = set_mode_info('prestretch1D')[1]
        color_isotropic = set_mode_info('isotropic')[1]
    else:
        color_plane = 'lightgray'
        color_uniaxial = 'lightgray'
        color_biaxial = 'lightgray'
        color_surface = 'lightgray'
        color_prestretch1D = 'lightgray'
        color_isotropic = 'lightgray'

    # Cao & Hutchinson 2012 (numerical, plane compression)
    plt.semilogx(5., 0.175, color=color_plane, markeredgecolor='k', marker='*', markersize=10)
    plt.semilogx(30., 0.055, color=color_plane, markeredgecolor='k', marker='*', markersize=10)

    # Brau et al. 2011 (experimental, plane)
    plt.semilogx(120., 0.17, color=color_plane, markeredgecolor='k', marker='*', markersize=10)

    # Auguste et al. 2014 (experimental)
    plt.semilogx(175., 0.04, color=color_uniaxial, markeredgecolor='k', marker='*', markersize=10)

    # Jin et al. 2015 (experimental, uniaxial subcritical creases)
    plt.semilogx(2., 0.370, color=color_uniaxial, markeredgecolor='k', marker='^', markersize=8)
    # Jin et al. 2015 (experimental, uniaxial wrinkles)
    plt.semilogx(4., 0.210, color=color_uniaxial, markeredgecolor='k', marker='*', markersize=10)
    # Jin et al. 2015 (numerical, subcritical crease)
    plt.semilogx(1.4, 0.36, color=color_plane, markeredgecolor='k', marker='^', markersize=8)

    # Hong et al. 2009 (numeric, plane creasing)
    plt.semilogx(1., 0.35, color=color_plane, markeredgecolor='k', marker='^', markersize=8)
    # Hong et al. 2009 (numeric, uniaxial creasing)
    plt.semilogx(1., 0.436, color=color_uniaxial, markeredgecolor='k', marker='^', markersize=8)
    # Hong et al. 2009 (numeric, biaxial creasing)
    plt.semilogx(1., 0.249, color=color_biaxial, markeredgecolor='k', marker='^', markersize=8)

    # Huang et al. 2005 (numerical, biaxial)
    plt.semilogx(325., 0.0089, color=color_biaxial, markeredgecolor='k', marker='*', markersize=10)

    # Zang et al. 2012 (wrinkling, numeric)
    plt.semilogx(836., 0.01, color=color_plane, markeredgecolor='k', marker='*', markersize=10)

    # Tallinen et al. 2014 (numeric, surface)
    plt.semilogx(1., 0.224, color=color_surface, markeredgecolor='k', marker='^', markersize=8)

    # Tallinen & Biggins 2015 (numerical, prestretch)
    plt.semilogx(1.14, 0.354, color=color_prestretch1D, markeredgecolor='k', marker='^', markersize=8)

    # Tallinen & Biggins 2015 (numerical, prestretch)
    [betas, strains, wavelengths] = data_Tallinen2015()
    plt.semilogx(
        betas,
        strains,
        linestyle='',
        color=color_prestretch1D,
        markeredgecolor='k',
        marker='*',
        markersize=10)

    # Budday et al. 2015 (numerical)
    plt.semilogx(5., 0.245, color=color_isotropic, markeredgecolor='k', marker='*', markersize=10)
    plt.semilogx(8., 0.159, color=color_isotropic, markeredgecolor='k', marker='*', markersize=10)

    # Cai et al. 2011 (should be at 10,000)
    plt.semilogx(1000., 0.025, color=color_isotropic, markeredgecolor='k', marker='*', markersize=10)


def data_wavelength(figname, color=True):
    """ add wavelength datapoints to Fig 5B

    Parameters
    ----------
    figname : string
        name of figure for datapoints to be added to
    color : boolean (default = True)
        determines if datapoints should be added in their specific loadcolor, or all gray

    Returns
    -------
    None

    """

    plt.figure(figname)

    if color:
        color_biaxial = set_mode_info('biaxial')[1]
        color_surface = set_mode_info('surface')[1]
        color_prestretch2D = set_mode_info('prestretch2D')[1]
        color_isotropic = set_mode_info('isotropic')[1]
    else:
        color_biaxial = 'lightgray'
        color_surface = 'lightgray'
        color_prestretch2D = 'lightgray'
        color_isotropic = 'lightgray'

    # Huang et al. 2005 (numeric, biaxial)
    plt.semilogx(1000., 43.75, color=color_biaxial, markeredgecolor='k', marker='*', markersize=10)
    plt.semilogx(325., 29.16, color=color_biaxial, markeredgecolor='k', marker='*', markersize=10)

    # Tallinen et al. 2014 (numeric, surface)
    plt.semilogx(1., 4.36, color=color_surface, markeredgecolor='k', marker='*', markersize=10)

    # Tallinen & Biggins 2015 (numerical, prestretch)
    [betas, strains, wavelengths] = data_Tallinen2015()
    plt.semilogx(
        betas,
        wavelengths,
        linestyle='',
        color=color_prestretch2D,
        markeredgecolor='k',
        marker='*',
        markersize=10)

    # Budday et al. 2014
    plt.semilogx(3., 10., color=color_isotropic, markeredgecolor='k', marker='*', markersize=10)

    # Sultan & Boudaoud 2008 (experimental, isotropic growth)
    plt.semilogx(3.4, 5.34, color=color_isotropic, markeredgecolor='k', marker='^', markersize=8)


def data_eff_strain():
    """ add strain datapoints to Fig 7A

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    plt.figure('7A')

    # Cao & Hutchinson 2012 (numerical, plane compression)
    plot_eff_strain_data([5.], [0.175], 'plane', 'wrinkling')
    plot_eff_strain_data([30.], [0.055], 'plane', 'wrinkling')

    # Brau et al. 2011 (experimental, plane)
    plot_eff_strain_data([120.], [0.17], 'plane', 'wrinkling')

    # Auguste et al. 2014 (experimental)
    plot_eff_strain_data([175.], [0.04], 'uniaxial', 'wrinkling')

    # Jin et al. 2015 (numerical, subcritical crease)
    plot_eff_strain_data([1.4], [0.36], 'plane', 'cusping')
    # Jin et al. 2015 (experimental, uniaxial subcritical creases)
    plot_eff_strain_data([2.], [0.37], 'uniaxial', 'cusping')
    # Jin et al. 2015 (experimental, uniaxial wrinkles)
    plot_eff_strain_data([4.], [0.21], 'uniaxial', 'wrinkling')

    # Hong et al. 2009 (numeric, plane creasing)
    plot_eff_strain_data([1.], [0.35], 'plane', 'cusping')
    # Hong et al. 2009 (numeric, uniaxial creasing)
    plot_eff_strain_data([1.], [0.436], 'uniaxial', 'cusping')
    # Hong et al. 2009 (numeric, biaxial creasing)
    plot_eff_strain_data([1.], [0.249], 'biaxial', 'cusping')

    # Huang et al. 2005 (numerical, biaxial) 
    plot_eff_strain_data([325.], [0.0089], 'biaxial', 'wrinkling')

    # Zang et al. 2012 (wrinkling, numeric)
    plot_eff_strain_data([836.], [0.01], 'plane', 'wrinkling')

    # Tallinen et al. 2014 (numeric, surface)
    plot_eff_strain_data([1.], [0.224], 'surface', 'cusping')

    # Tallinen & Biggins 2015 (numerical, prestretch)
    plot_eff_strain_data([1.14], [0.354], 'prestretch1D', 'cusping')
    [betas, strains, wavelengths] = data_Tallinen2015()
    plot_eff_strain_data(betas, strains, 'prestretch2D', 'wrinkling')

    # Budday et al. 2015 (numerical)
    plot_eff_strain_data([5.], [0.245], 'isotropic', 'wrinkling')
    plot_eff_strain_data([8.], [0.159], 'isotropic', 'wrinkling')

    # Cai et al. 2011 (should be at b=10,000)
    plot_eff_strain_data([1000.], [0.025], 'isotropic', 'wrinkling')


def data_eff_wavelength():
    """ add wavelength datapoints to Fig 7B

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    plt.figure('7B')

    # Huang et al. 2005 (numeric, biaxial)
    plot_eff_wavelength_data([325.], [29.16], [0.0089], 'biaxial', 'wrinkling')

    # Tallinen et al. 2014 (numeric, surface)
    plot_eff_wavelength_data([1.], [4.36], [0.224], 'isotropic', 'wrinkling')

    # Tallinen & Biggins 2015 (numerical, prestretch)
    [betas, strains, wavelengths] = data_Tallinen2015()
    plot_eff_wavelength_data(betas, wavelengths, strains, 'prestretch2D', 'wrinkling')


def plot_curves(mode, figno, subfigno):
    """ find the min (and max) critical min_strains, and corresponding wavelengths and stiffness ratios

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    figno : int
        figure number corresponding to paper
    subfigno : int
        subfigure number

    Returns
    -------
    None

    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    with open('data_betas.txt', 'r') as file_beta:
        beta_strings = file_beta.readlines()
    with open('data_strain_{mode}.txt'.format(mode=mode)) as file_strain:
        strain_strings = file_strain.readlines()
    with open('data_wavelength_{mode}.txt'.format(mode=mode)) as file_wavelength:
        wavelength_strings = file_wavelength.readlines()

    plt.figure('{figno}_{subfigno}_{mode}'.format(figno=figno, subfigno=subfigno, mode=mode))
    # plt.xlabel('normalized wavenumber $L/H$')
    # plt.ylabel('critical axial strain $\\epsilon_1$')
    plt.gca().set_ylim(0., 1.)
    plt.gca().set_xlim(0.1, 1000.)

    for i in range(len(beta_strings)):
        beta = float(beta_strings[i])

        if beta in [0.1, 10., 100., 1000.]:
            width = 2
        elif beta in [1.]:
            width = 4
        else:
            width = 1

        wavelengths_all = [float(x) for x in wavelength_strings[i].split()]
        strains_all = [float(x) for x in strain_strings[i].split()]

        plt.semilogx(wavelengths_all, strains_all, color=loadcolor, linestyle='-', linewidth=width)


def plot_wavelengths(mode, figno, subfigno):
    """ Figs. 5B, 10B

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    figno : int
        figure number corresponding to paper
    subfigno : int
        subfigure number

    Returns
    -------
    min_wavelengths : list of floats
        list of wavelengths associated with minimum min_strains
    min_strains : list of floats
        list of minimum strain values

    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    with open('data_min_{mode}.txt'.format(mode=mode), 'r') as file_min:
        min_strings = file_min.readlines()

    min_betas = numpy.zeros(len(min_strings))
    min_strains = numpy.zeros(len(min_strings))
    min_wavelengths = numpy.zeros(len(min_strings))
    for i in range(len(min_strings)):
        [min_betas[i], min_strains[i], min_wavelengths[i]] = [float(x) for x in min_strings[i].split()]

    with open('data_max_{mode}.txt'.format(mode=mode), 'r') as file_max:
        max_strings = file_max.readlines()

    max_betas = numpy.zeros(len(max_strings))
    max_strains = numpy.zeros(len(max_strings))
    max_wavelengths = numpy.zeros(len(max_strings))
    for i in range(len(max_strings)):
        [max_betas[i], max_strains[i], max_wavelengths[i]] = [float(x) for x in max_strings[i].split()]

    start = 0
    # remove zero-wavelength results
    for i in range(len(min_wavelengths)):
        if min_wavelengths[i] == 0.0:
            start = i + 1
        else:
            break

    betas_nonzero = numpy.array(min_betas[start:])
    wavelengths_nonzero = numpy.array(min_wavelengths[start:])
    crit_strains_nonzero = numpy.array(min_strains[start:])

    # plot threshold values on curve plots     
    plt.figure('{figno}_{subfigno}_{mode}'.format(figno=figno, subfigno=subfigno, mode=mode))
    plt.semilogx(wavelengths_nonzero, crit_strains_nonzero, color='k', linestyle='-', linewidth=4, zorder=100)

    # add zero-wavelength and max threshold values as dotted lines
    if mode_type != 'FvK':
        dotted_strains = numpy.append(max_strains, crit_strains_nonzero[0])
        dotted_wavelengths = numpy.append(max_wavelengths, wavelengths_nonzero[0])
        plt.semilogx(dotted_wavelengths, dotted_strains, color='k', linestyle='--', linewidth=2, zorder=100)

    if mode == 'prestretch2D':
        data_Tallinen2015(figno, plot=True)

    # plt.xlabel('normalized wavelength')
    # plt.ylabel('critical strain')
    plt.savefig('{figno}_{subfigno}_{mode}.pdf'.format(figno=figno, subfigno=subfigno, mode=mode))

    # slightly offset some curves so that they all show on the plot
    if mode in ['biaxial']:
        wavelengths_nonzero = wavelengths_nonzero + 0.5
    elif mode in ['plane']:
        wavelengths_nonzero = wavelengths_nonzero - 0.5

    if mode_type == 'compression':
        width = 3
    else:
        width = 4

    plt.figure('5B')
    plt.semilogx(
        betas_nonzero,
        wavelengths_nonzero,
        color=loadcolor,
        linestyle='-',
        linewidth=width,
        label=mode + ' ' + mode_type
    )

    if mode_type == 'compression':
        all_wavelengths = numpy.append(max_wavelengths, wavelengths_nonzero[0])
        all_betas = numpy.append(max_betas, betas_nonzero[0])
        plt.semilogx(all_betas, all_wavelengths, color=loadcolor, linestyle='-', linewidth=1)

    plt.figure('10B')
    plt.semilogx(
        betas_nonzero,
        wavelengths_nonzero,
        color='lightgray',
        linestyle='-',
        linewidth=3,
        label=mode + ' ' + mode_type
    )

    if mode_type == 'compression':
        plt.semilogx(all_betas, all_wavelengths, color='lightgray', linestyle='-', linewidth=1)

    return min_wavelengths, min_strains


def plot_strains(mode, min_strains):
    """ Figs. 5A and 8

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    min_strains : list of floats
        list of min_strains associated with each stiffness ratio

    Returns
    -------
    None

    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    with open('data_betas.txt', 'r') as file_beta:
        beta_strings = file_beta.readlines()
    betas = [float(x) for x in beta_strings]

    plt.figure('5A')  # critical min_strains
    # slightly offset some curves so that they all show on the plot
    if mode in ['surface', 'unidirectional']:
        crit_strains_adjusted = numpy.array(min_strains) + 0.01
        width = 3
    elif mode in ['isotropic']:
        crit_strains_adjusted = numpy.array(min_strains) - 0.01
        width = 3
    else:
        crit_strains_adjusted = numpy.array(min_strains)
        width = 4
    plt.semilogx(
        betas,
        crit_strains_adjusted,
        color=loadcolor,
        linestyle='-',
        linewidth=width,
        label=mode + ' ' + mode_type
    )

    plt.figure('10A')  # critical min_strains
    plt.semilogx(betas, min_strains, color='lightgray', linestyle='-', linewidth=3, label=mode + ' ' + mode_type)

    if mode_type != 'FvK':
        plt.figure('8A')  # critical axial pressure
        crit_pressure = calc_axial_pressure(mode, min_strains)
        plt.semilogx(betas, crit_pressure, color=loadcolor, linestyle='-', linewidth=4, label=mode + ' ' + mode_type)

        plt.figure('8B')  # critical hydrostatic pressure
        crit_pressure = calc_hydro_pressure(mode, min_strains)
        plt.semilogx(betas, crit_pressure, color=loadcolor, linestyle='-', linewidth=4, label=mode + ' ' + mode_type)


def plot_eff_wavelength_strain(mode, min_wavelengths, min_strains, n):
    """ Fig. 7

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    betas : list of floats
        list of stiffness ratios (film/substrate) 
    min_wavelengths : list of floats
        list of wavelengths associated with each stiffness ratio
    min_strains : list of floats
        list of min_strains associated with each stiffness ratio

    Returns
    -------
    None
    
    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    with open('data_betas.txt', 'r') as file_beta:
        beta_strings = file_beta.readlines()
    betas = [float(x) for x in beta_strings]

    wavelength_eff = calc_eff_wavelength(mode_type, functions, min_strains, min_wavelengths)
    beta_eff = calc_eff_stiffness(mode_type, functions, min_strains, betas)

    width = 3

    plt.figure('7B')
    # slightly offset some curves so that they all show on the plot (plane, prestretch1D are left untouched)
    if mode in ['uniaxial', 'prestretch2D']:
        wavelength_eff = wavelength_eff + 0.4
    elif mode in ['biaxial', 'unidirectional']:
        wavelength_eff = wavelength_eff + 0.8
    elif mode in ['surface']:
        wavelength_eff = wavelength_eff + 1.2
    elif mode in ['isotropic']:
        wavelength_eff = wavelength_eff + 1.6

    beta_eff_plot = numpy.insert(beta_eff, 0, 0.1)
    wavelength_eff_plot = numpy.insert(wavelength_eff, 0, wavelength_eff[0])
    plt.semilogx(
        beta_eff_plot,
        wavelength_eff_plot,
        color=loadcolor,
        linestyle='-',
        linewidth=width,
        label=mode + ' ' + mode_type
    )

    plt.figure('7A')  # effective min_strains and stiffness
    beta_eff = calc_eff_stiffness(mode_type, functions, min_strains, betas)
    strains_eff = calc_eff_strain(mode_type, functions, min_strains)
    beta_eff_plot = numpy.insert(beta_eff, 0, 0.1)
    strains_eff_plot = numpy.insert(strains_eff, 0, strains_eff[0])
    plt.semilogx(
        beta_eff_plot,
        strains_eff_plot + (n - 4.) / 150.,
        color=loadcolor,
        linestyle='-',
        linewidth=width,
        label=mode + ' ' + mode_type
    )


def plot_effective_measures(mode):
    """ Fig. 6

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'

    Returns
    -------
    None

    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    lams = numpy.linspace(0.1, 1.0, 50)
    length = 0.015
    width = 4

    if mode_type == 'growth':
        [g_1, g_2, g_3] = functions
        [d_1, d_3] = [d_one, d_one]
    else:  # whole-domain compression does not affect stiffness ratio
        [g_1, g_2, g_3] = [d_one, d_one, d_one]
        [d_1, d_3] = functions

    beta_eff = numpy.zeros(len(lams))
    wave_eff = numpy.zeros(len(lams))
    strain_eff = numpy.zeros(len(lams))

    for i in range(len(lams)):
        lam = lams[i]
        d1 = d_1(lam)
        d3 = d_3(lam)
        g1 = g_1(lam)
        g2 = g_2(lam)
        g3 = g_3(lam)
        lam1 = d1 / g1
        lam3 = d3 / g3

        if mode_type == 'growth':
            beta_eff[i] = (lam1 ** 2. + 1. / (lam1 * lam3) ** 2.) / 2.
        else:
            beta_eff[i] = 1.

        wave_eff[i] = d1 ** 2. * d3 / (g1 * g2 * g3)
        strain_eff[i] = 1. - lam1 * lam3 ** 0.5

    plt.figure('6B')  # curved effective stiffness lines
    if mode in ['unidirectional', 'surface']:
        offset = 0.6 * length
    elif mode in ['isotropic']:
        offset = 1.2 * length
    else:
        offset = 0. * length
    strains = 1. - lams + offset
    if mode == 'plane':
        beta_eff = beta_eff - 0.03
    elif mode == 'biaxial':
        beta_eff = beta_eff + 0.02
    if mode_type == 'compression':
        width = 2
    else:
        width = 3

    plt.semilogy(strains, beta_eff, color=loadcolor, marker='', linestyle='-', linewidth=width)
    plt.gca().set_ylim(0.8, 100.)
    plt.gca().set_xlim(0., 1.)
    # plt.xlabel('axial strain')
    # plt.ylabel('effective stiffness')
    plt.savefig('6B_effective-stiffness.pdf')

    plt.figure('6C')  # curved effective wavelength lines
    if mode in ['isotropic', 'surface']:
        offset = 0.8 * length
    else:
        offset = 0. * length
    strains = 1. - lams + offset
    if mode == 'prestretch2D':
        wave_eff = wave_eff + 0.008
    if mode in ['prestretch1D', 'prestretch2D']:
        width = 3
    else:
        width = 4

    plt.plot(strains, wave_eff, color=loadcolor, marker='', linestyle='-', linewidth=width)
    plt.gca().set_xlim(0., 1.0)
    plt.gca().set_ylim(0., 1.2)
    # plt.xlabel('axial strain')
    # plt.ylabel('effective wavelength')
    plt.savefig('6C_effective-wavelength.pdf')

    plt.figure('6A')  # curved effective strain lines
    width = 4
    if mode in ['prestretch1D', 'prestretch2D']:
        offset = 1. * length
    elif mode in ['unidirectional', 'surface', 'isotropic']:
        offset = 2. * length
    else:
        offset = 0. * length
    strains = 1. - lams + offset

    plt.plot(strains, strain_eff, color=loadcolor, marker='', linestyle='-', linewidth=width)
    plt.gca().set_xlim(0., 1.)
    plt.gca().set_ylim(0., 1.)
    # plt.xlabel('axial strain')
    # plt.ylabel('effective wavelength')
    plt.savefig('6A_effective-strain.pdf')


def plot_eff_strain_data(betas, strains, mode, instability):
    """ Add datapoints to Fig. 7A

    Parameters
    ----------
    betas : list of floats
        list of stiffness ratios (film/substrate) 
    strains : list of floats
        list of strain values
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    instability : string ('wrinkling' or 'cusping')
        type of instability associated with the data point

    Returns
    -------

    Notes
    -----
    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    if instability == 'wrinkling':
        symbol = '*'
        size = 10
    elif instability == 'cusping':
        symbol = '^'
        size = 8

    B_eff = calc_eff_stiffness(mode_type, functions, strains, betas)
    e_eff = calc_eff_strain(mode_type, functions, strains)

    plt.figure('7A')
    plt.semilogx(B_eff, e_eff, linestyle='', color=loadcolor, markeredgecolor='k', marker=symbol, markersize=size)


def plot_eff_wavelength_data(betas, wavelengths, strains, mode, instability):
    """ 

    Parameters
    ----------
    betas : list of floats
        list of stiffness ratios (film/substrate) 
    wavelengths : list of floats
        list of wavelength values
    strains : list of floats
        list of strain values
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    instability : string ('wrinkling' or 'cusping')
        type of instability associated with the data point

    Returns
    -------
    None

    """

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    if instability == 'wrinkling':
        symbol = '*'
        size = 10
    elif instability == 'cusping':
        symbol = '^'
        size = 8

    B_eff = calc_eff_stiffness(mode_type, functions, strains, betas)
    lh_eff = calc_eff_wavelength(mode_type, functions, strains, wavelengths)

    plt.figure('7B')
    plt.semilogx(B_eff, lh_eff, linestyle='', color=loadcolor, markeredgecolor='k', marker=symbol, markersize=size)


def plot_FvK_curves(FvK_mode, mode, figno):
    """ plot critical strains, and corresponding wavelengths for FvK equations

    Parameters
    ----------
    FvK_mode : string
        type of FvK loading
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    figno : int
        figure number

    Returns
    -------
    None

    """

    [mode_type, loadcolor, functions] = set_mode_info(FvK_mode)

    with open('data_min_{FvK_mode}.txt'.format(FvK_mode=FvK_mode), 'r') as file_min:
        min_strings = file_min.readlines()
    min_betas = numpy.zeros(len(min_strings))
    min_strains = numpy.zeros(len(min_strings))
    min_wavelengths = numpy.zeros(len(min_strings))
    for i in range(len(min_strings)):
        [min_betas[i], min_strains[i], min_wavelengths[i]] = [float(x) for x in min_strings[i].split()]

    # plot threshold values on curve plots     
    plt.figure('9_{figno}_{mode}'.format(figno=figno, mode=mode))

    if 'original' in FvK_mode:
        linestyle = '--'
        loadcolor = 'gray'
    else:
        linestyle = '-'

    plt.semilogx(min_wavelengths, min_strains, color=loadcolor, linestyle=linestyle, linewidth=2, zorder=10)

    plt.savefig('9_{figno}_{mode}.pdf'.format(figno=figno, mode=mode))


def plot_FvK_strains_wavelengths(FvK_mode):
    """ plot critical strains for FvK equations on Fig. 10A and critical wavelengths on Fig. 10B

    Parameters
    ----------
    FvK_mode : string
        type of FvK loading

    Returns
    -------
    None

    """

    [mode_type, loadcolor, functions] = set_mode_info(FvK_mode)

    if 'original' in FvK_mode:
        loadcolor = 'k'
        linestyle = '--'
    else:
        linestyle = '-'

    with open('data_min_{FvK_mode}.txt'.format(FvK_mode=FvK_mode), 'r') as file_min:
        min_strings = file_min.readlines()

    min_betas = numpy.zeros(len(min_strings))
    min_strains = numpy.zeros(len(min_strings))
    min_wavelengths = numpy.zeros(len(min_strings))
    for i in range(len(min_strings)):
        [min_betas[i], min_strains[i], min_wavelengths[i]] = [float(x) for x in min_strings[i].split()]

    # plot threshold values on curve plots     
    plt.figure('10A')
    plt.semilogx(min_betas, min_strains, color=loadcolor, linestyle=linestyle, linewidth=4, zorder=100)

    plt.figure('10B')
    plt.semilogx(min_betas, min_wavelengths, color=loadcolor, linestyle=linestyle, linewidth=4, zorder=100)


def save_figures():
    """ Save figures

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    plt.figure('5B')
    plt.gca().xaxis.grid(b=True, which='major', color='k', linestyle='--')
    plt.gca().set_xlim(0.1, 1000)
    plt.gca().set_ylim(0., 45)
    # plt.title('Buckling Wavelengths for Different Stiffness Ratios')
    # plt.xlabel('stiffness ratio $\\beta$')
    # plt.ylabel('normalized wavenumber $L/H$')
    # plt.legend(loc='best')
    plt.savefig('5B_critical-wavelengths.pdf')

    plt.figure('5A')
    plt.gca().xaxis.grid(b=True, which='major', color='k', linestyle='--')
    plt.gca().set_xlim(0.1, 1000)
    plt.gca().set_ylim(0., 1.)
    # plt.title('Critical Strains for Different Stiffness Ratios')
    # plt.xlabel('stiffness ratio $\\beta$')
    # plt.ylabel('critical axial strain $\\epsilon_1$')
    # plt.legend(loc='best')
    plt.savefig('5A_critical-min_strains.pdf')

    plt.figure('7B')
    # plt.xlabel('effective stiffness ratio $\\beta_{eff}$')
    # plt.ylabel('effective wavelength l/h $')
    plt.gca().xaxis.grid(b=True, which='major', color='k', linestyle='--')
    plt.gca().set_xlim(0.1, 1000)
    plt.gca().set_ylim(0., 45.)
    plt.savefig('7B_effective-wavelengths.pdf')

    plt.figure('7A')
    # plt.xlabel('effective stiffness ratio $\\beta_{eff}$')
    # plt.ylabel('critical effective strain $\\epsilon_{eff}$')
    plt.gca().xaxis.grid(b=True, which='major', color='k', linestyle='--')
    plt.gca().set_xlim(0.1, 1000)
    plt.gca().set_ylim(0., 1.)
    plt.axhline(y=0.35, color='k', linestyle='-', linewidth=2)  # creasing
    plt.axhline(y=0.4563, color='k', linestyle='--', linewidth=2)  # Biot
    plt.savefig('7A_effective-min_strains.pdf')

    plt.figure('8A')
    plt.gca().set_ylim(0., 5.)
    # plt.title('Critical Axial Pressures for Different Stiffness Ratios')
    # plt.xlabel('stiffness ratio $\\beta$')
    # plt.ylabel('normalized critical axial pressure $-\\sigma_{11} / \\mu_s$')
    # plt.legend(loc='best')
    plt.savefig('8A_critical-pressure-axial.pdf')

    plt.figure('8B')
    plt.gca().set_ylim(0., 3.5)
    # plt.title('Critical Hydrostatic Pressures for Different Stiffness Ratios')
    # plt.xlabel('stiffness ratio $\\beta$')
    # plt.ylabel('normalized critical pressure $- tr \\sigma /(3 \\mu_s)$')
    # plt.legend(loc='best')
    plt.savefig('8B_critical-pressure-hydro.pdf')

    plt.figure('10A')
    plt.gca().set_ylim(0.0, 1.0)
    plt.savefig('10A_FvK_strains.pdf')

    plt.figure('10B')
    plt.gca().set_ylim(0.0, 45.0)
    plt.savefig('10B_FvK_wavelengths.pdf')


def write_results(mode, min_strains, min_wavelengths):
    """ Write results for given mode_type of loading

    Parameters
    ----------
    mode : string
        type of loading, from 'prestretch1D', 'prestretch2D', 'unidirectional' , 'surface', 'isotropic', 'plane', 'uniaxial', 'biaxial', 'compression', 'growth'
    min_strains : list of floats
        list of critical strain values for given values of beta
    min_wavelengths : list of floats
        list of wavelengths values for given values of beta

    Returns
    -------
    None

    Notes
    -----
    Is this being called?  Where is this plotting?

    """

    f = open('results.txt', 'a')

    [mode_type, loadcolor, functions] = set_mode_info(mode)

    with open('data_betas.txt', 'r') as file_beta:
        beta_strings = file_beta.readlines()
    betas = [float(x) for x in beta_strings]

    f.write("%s \n" % mode)

    # find b_wrinkle, stiffness below which wrinkling instability is not energetically favorable
    for i in range(len(betas)):
        if min_wavelengths[i] != 0:
            b_wrinkle = betas[i - 1]
            break
    # plt.axvline(x=b_wrinkle,color=loadcolor,linestyle='--')

    # find critical strain for zero-wavelength instability
    strain_zw = min_strains[0]
    f.write("zero wavelength instability occurs at strain = {strain:0.4f}, at and below b = {beta:0.4f} \n".format(
        strain=strain_zw,
        beta=b_wrinkle
    ))

    # find stiffness at which infinitesimal cusped sulci occurs first (from Mahdevan & Hohlfield, via Tallinen)
    stretch_sulci = 0.647
    strain_sulci = 1. - stretch_sulci
    plt.axhline(y=strain_sulci, color='k', linestyle='-')

    eff_crit_strains = calc_eff_strain(mode_type, functions, min_strains)

    for i in range(len(min_strains)):
        if eff_crit_strains[i] < strain_sulci and i != 0:
            b_sulci = (betas[i] + betas[i - 1]) / 2.
            # plt.axvline(x=b_sulci, ymax=strain_sulci, color=loadcolor, linestyle='-.')
            break

    f.write("infinitesimal cusped sulci occur at strain = {strain:0.4f}, at and below b = {beta:0.4f} \n\n".format(
        strain=min_strains[i],
        beta=b_sulci
    ))

    f.close()
