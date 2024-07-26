import matplotlib.backends.backend_pdf
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from colour_schemes import hex


def save_figs_to_single_pdf(figlist, pdf_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
    for fig in figlist:
        pdf.savefig(fig)
    pdf.close()



def subplots_hide_xaxes(ax, nrows, ncols, keep_lowest=True, keep=[]):
    if keep_lowest:
        keep.append(nrows-1)

    for irow in range(nrows):
        for icol in range(ncols):
            if irow in keep:
                pass
            else:
                ax[irow, icol].spines['bottom'].set_visible(False)
                ax[irow, icol].set_xticks([])



def setup_we_mpl(create_example=False):
    m = mpl.rcParams
    m["font.weight"]          = 'bold'

    m["axes.spines.right"]    = 'False'
    m["axes.spines.top"]      = 'False'

    m["axes.linewidth"]       = 1.0
    m["axes.labelweight"]     = 'bold'

    # Axis buffers
    m["axes.xmargin"]     = 0
    m["axes.ymargin"]     = 0.02

    # Legends
    m["figure.figsize"]       = (12,7)
    m["legend.framealpha"]   = 1.0
    m["legend.facecolor"]    = 'white'
    m["legend.edgecolor"]    = 'black'

    # Set my custom default colourscheme currently Medium contrast
    # ignoring white and reverse order
    # I then do a leapfrog so that you cycle through a bit faster
    clrs = hex.Hexes().hex['MedContrast_PT'][1::][::-1]
    m["axes.prop_cycle"] = cycler(color=clrs[::2]+clrs[1::2])

    if create_example:
        fig, ax = plt.subplots()
        x = np.linspace(0, 2*np.pi, 1000)
        legs = []

        for i in range(5):
            ax.plot(x, np.sin(x*(i+1)))
            legs.append(f'line {i}')

        ax.set_title('Test title qwerty')
        ax.legend(legs)
        ax.set_xlabel('My x label')
        ax.set_ylabel('My y label')

        plt.show()
    return m

if __name__ == "__main__":
    setup_we_mpl(create_example=True)