from JMPS17_subroutines_calculate import *
from JMPS17_subroutines import *

warnings.simplefilter('ignore')

if __name__ == '__main__':

    ####################################################################################
    modes = ['biaxial', 'uniaxial', 'plane', 'prestretch1D', 'prestretch2D', 'unidirectional', 'surface', 'isotropic'] #

    for mode_type in ['FvK_compression ', 'FvK_growth ']:
        for modification in ['original', 'mod_thickness', 'mod_wavelength', 'mod_both']:
            modes.append(mode_type + modification)

    betas = numpy.logspace(-1., 3., 5)

    # parameters for root finding (wavelength = 2.*pi/kh = L/H)
    n_wavelengths = 1000
    wavelengths = numpy.logspace(-1., 3., num=n_wavelengths)
    wavelengths = wavelengths[::-1] # start at right end of the graph (where the distance between roots is larger)
    tol = 1.e-12 # this is a default value

    # parameters for output
    findroots = True        # only set to false for troubleshooting, using plotroots below
    plotroots = False       # save plot of absolute value of determinant at each n_wavelengths
    plotindcurves = False   # save plot of individual curves for each beta
    printoutput = False     # print every root found at every n_wavelengths

    ####################################################################################

    numpy.savetxt('data_wavelengths.txt', wavelengths, fmt='%.4f')
    numpy.savetxt('data_betas.txt', betas, fmt='%.4f')

    for n in range(len(modes)):

        mode = modes[n]
        [mode_type, loadcolor, functions] = set_mode_info(mode)

        mode_data_strain = open('data_strain_{mode}.txt'.format(mode=mode), 'w')
        mode_data_wavelength = open('data_wavelength_{mode}.txt'.format(mode=mode), 'w')

        # parameters for looking for the existence of roots
        npts = 100
        lam_max = 1.1
        if mode in ['unidirectional', 'prestretch1D', 'prestretch2D'] or mode_type == 'FvK':
            lam_min = 0.01
        else:
            lam_min = 0.1

        min_wavelengths = []
        max_wavelengths = []
        min_strains = []
        max_strains = []
        min_betas = []
        max_betas = []

        for j in range(len(betas)):

            beta = betas[j]

            [strains_all, wavelengths_all] = find_critical_values(mode, beta, wavelengths, lam_min, lam_max, npts, plotroots, findroots, printoutput, plotindcurves, tol)
            write_crit_values(mode_data_strain, strains_all)
            write_crit_values(mode_data_wavelength, wavelengths_all)

            [min_strains, min_wavelengths, min_betas, max_strains, max_wavelengths, max_betas] = find_threshold_values(beta, mode, strains_all, wavelengths_all, min_strains, min_wavelengths, min_betas, max_strains, max_wavelengths, max_betas, findroots)

        mode_data_strain.close()
        mode_data_wavelength.close()
        write_threshold_values('data_min_{mode}.txt'.format(mode=mode), min_strains, min_wavelengths, min_betas)
        write_threshold_values('data_max_{mode}.txt'.format(mode=mode), max_strains, max_wavelengths, max_betas)

        # plt.show()

