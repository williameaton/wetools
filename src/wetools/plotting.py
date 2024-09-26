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
    m["axes.xmargin"]     = 0.0
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
        fig, ax = plt.subplots(2,2)
        x = np.linspace(0, 2*np.pi, 1000)
        legs = []

        for i in range(5):
            ax[0,0].plot(x, np.sin(x*(i+1)))
            legs.append(f'line {i}')

        ax[0,0].set_title('Test title qwerty')
        ax[0,0].legend(legs)
        ax[0,0].set_xlabel('My x label')
        ax[0,0].set_ylabel('My y label')

        # Hist
        d = np.random.random(50)
        ax[0,1].hist(d)

        h = np.random.random(50)
        ax[1, 0].scatter(d,np.sin(h**2) + 7)

        # Imshow
        mat =  np.random.random((30, 30))
        ax[1, 1].imshow(mat)



        we_adjust_spines(fig)

        plt.show()
    return m


def align_y_labels(ax, nrows, coord=-0.2):
    for i in range(nrows):
        ax[i, 0].yaxis.set_label_coords(coord, 0.5)



def we_adjust_spines(fig):
    axes = fig.get_axes()

    for ax in axes:
        if (len(ax.lines)==0 and len(ax.patches)==0 and len(ax.images)!=0):
            pass
        else :
            # Default values of xmin and xmax
            xlim = ax.get_xlim()

            """
            # Sweep through the different types of artists that may be on there
            lines   = ax.lines
            imgs    = ax.images
            patches = ax.patches
    
    
            for l in lines:
                ld = l.get_data()
                l_x = ld[0]
                l_y = ld[1]
    
                #l_max_x = max(l_x)
                #if(l_max_x > x_max):
                #    x_max = l_max_x
                l_min_x = min(l_x)
                if (l_min_x < x_min):
                    x_min = l_min_x
    
            for p in patches:
                l_min_x, lmin_y = p.get_xy()
                p_w = p.get_width()
                p_h = p.get_height()
    
                if (l_min_x < x_min):
                    x_min = l_min_x"""

            ticks = ax.get_xticks()
            el_ticks = ticks[ticks <= xlim[0]]
            # Minimum tick that is above the original xlimit
            min_t = max(el_ticks)

            # Create new xlimits so that there is a constant buffer (3 %)
            buf = (xlim[1]  - min_t) * 0.03
            new_len = (xlim[1]  - min_t) + buf
            ax.set_xlim([xlim[1]-new_len,   xlim[1]+buf ])

            ax.spines.bottom.set_bounds(min_t,  min([xlim[1]+buf, max(ticks)] ))


if __name__ == "__main__":
    setup_we_mpl(create_example=True)