import matplotlib.backends.backend_pdf

def save_figs_to_single_pdf(figlist, pdf_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
    for fig in figlist:
        pdf.savefig(fig)
    pdf.close()
