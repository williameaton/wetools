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

def setup_we_mpl(create_example=False):
    m = mpl.rcParams
    m["font.weight"]          = 'bold'
    m["axes.spines.right"]    = 'False'
    m["axes.spines.top"]      = 'False'
    m["axes.linewidth"]       = 1.0
    m["axes.labelweight"]     = 'normal'
    m["figure.figsize"]       = (12,7)

    # Set my custom default colourscheme currently Medium contrast
    # ignoring white and reverse order
    clrs = hex.Hexes().hex['MedContrast_PT'][1:]
    clrs = clrs[::-1]
    m["axes.prop_cycle"] = cycler(color=clrs)

    if create_example:
        fig, ax = plt.subplots(2,2)
        x = np.linspace(0, 2*np.pi, 1000)
        for i in range(6):
            ax[0,0].plot(x, np.sin(x*(i+1)))
        plt.show()
    return m


