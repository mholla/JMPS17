from JMPS17_subroutines_plot import *

warnings.simplefilter('ignore')

if __name__ == '__main__':

    ####################################################################################
    modes = ['biaxial', 'uniaxial', 'plane',
             'prestretch1D', 'prestretch2D',
             'unidirectional', 'surface', 'isotropic']

    FvK_modes = []
    for mode_type in ['FvK_compression ', 'FvK_growth ']:
        for modification in ['mod_both', 'original']:
            FvK_modes.append(mode_type + modification)

    # plot options
    font = {'size': 16}
    plt.rc('font', **font)

    # create new results file
    f = open('results.txt', 'w')
    f.close()

    ####################################################################################

    # 8 modes of loading
    for n in range(len(modes)):
        mode = modes[n]
        [mode_type, loadcolor, functions] = set_mode_info(mode)

        # Figs. 4, 5, 8
        plot_curves(mode, 4, n)
        [min_wavelengths, min_strains] = plot_wavelengths(mode, 4, n)
        plot_strains(mode, min_strains)

        # Fig. 6
        plot_effective_measures(mode)

        # Fig. 7
        plot_eff_wavelength_strain(mode, min_wavelengths, min_strains, n)

        write_results(mode, min_strains, min_wavelengths)

    # Foppl-van Karman equations
    for m in range(len(FvK_modes)):
        FvK_mode = FvK_modes[m]

        for n in range(len(modes)):
            mode = modes[n]

            # Fig. 9
            plot_curves(mode, 9, n)
            plot_wavelengths(mode, 9, n)
            plot_FvK_curves(FvK_mode, mode, n)

        # Fig. 10        
        plot_FvK_strains_wavelengths(FvK_mode)

    # add Cai & Fu analytical equation
    betas = numpy.logspace(-1., 3., 101)
    other_analytical(betas)

    # add datapoints to figures
    datapoints()

    save_figures()
