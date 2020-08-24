import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plotParams = {'mathtext.fontset':'stixsans',
              'font.family':'Minion Pro',
              'font.size':11,
              'figure.dpi':141,
              'lines.linewidth':1.0,
              'figure.subplot.left':.25,
              'figure.subplot.right':.75,
              'figure.subplot.bottom':.2,
              'figure.subplot.top':.9,
              'figure.subplot.wspace':.1,
              'figure.constrained_layout.wspace':.1,
              'figure.figsize':(6,2.3)}
plt.rcParams.update(plotParams)


def heatmap(results, params={}, c_range=None, bar=False, filter_color=None):
    """This function plots the results from the polynomial experiment notebook."""
    vmin, vmax = c_range
    kwargs = {'y_marks':4, 'x_pad':10, 'x_space':40, 'ax':None}
    kwargs.update(params)

    if kwargs['ax'] is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = plt.gcf()
        ax = kwargs['ax']

    cmap = emphasis(color=filter_color)
    im = ax.imshow(results, vmin=vmin, vmax=vmax, origin='lower', resample=True,
                interpolation='bicubic', cmap=cmap, alpha=.95,)

    if  bar:
        cax = fig.add_axes([.7,.35,.03,.4])
        ax.figure.colorbar(im, cax=cax, ticks=[0.2, 0.1, 0, -0.1, -0.2])

    if 'x_params' in kwargs and 'y_params' in kwargs:
        x_params, y_params = kwargs['x_params'], kwargs['y_params']
        x_0, x_lim = x_params
        y_res, y_lim = y_params
        y_marks, x_pad, x_space = kwargs['y_marks'], kwargs['x_pad'], kwargs['x_space']

        ax.set_xticks(range(x_0, x_lim-x_0, x_space))
        ax.set_xticklabels(range(x_0 + x_pad,x_lim-x_0, x_space))
        ax.set_yticks([i for i in range(y_res) if (i+1) % (y_res/y_marks) == 0])
        ax.set_yticklabels([y_lim/y_marks*i for i in range(1,y_marks+1)])    
        ax.set_xlabel(r'Sample size')
        ax.set_ylabel(r'Noise level $\sigma^2$ ~ N(0, $\sigma^2$)')
        # ax.set_title('Polynomial experiment')
        ax.axis('scaled')
    else:
        ax.axis('off')

    return fig

def emphasis(color, cmap='jet'):
    """This helper function fades out colors filtered by parameter *color*(rgb array)"""
    # preliminar arrangements
    old_cmap = plt.get_cmap(cmap)
    N = old_cmap.N
    cmap = old_cmap(range(N))

    if color is not None:
        # filtering emphasized color
        r, g, b = color.tolist()
        r_mask = cmap[:, 0] <= r
        g_mask = cmap[:, 1] <= g
        b_mask = cmap[:, 2] <= b

        # rgb_mask = (r_mask & g_mask) | (g_mask & b_mask) | (r_mask & b_mask)    
        rgb_mask = r_mask & g_mask & b_mask
        new_colors = cmap[rgb_mask]
        new_colors[:, -1] = .2
        cmap[rgb_mask] = new_colors

    return ListedColormap(cmap)

def plot_distribution(*data, bins=20):
    """This function plots distributions given as a list of dictionaries."""
    # define domain (x-axis)
    ends = np.vstack([dist['distribution'].ppf([.01,.90]) for dist in data])
    left, right = np.min(ends[:,0]), np.max(ends[:,1])
    bins = np.linspace(left, right, bins)
    xx = np.linspace(left, right, 300)

    #plotting
    fig, (pdf_ax, cdf_ax) = plt.subplots(1,2, constrained_layout=True, facecolor=[.5,.5,.8,.2])
    pdfParams = {'color':'C0', 'label':None, 'data':None}
    for dist in data:
        pdfParams.update(dist)
        pdf = dist['distribution'].pdf(xx)
        pdf_ax.fill_between(xx, pdf, color=pdfParams['color'], label=pdfParams['label'], alpha=.6)
        if dist.get('data') is not None:
            pdf_ax.hist(dist['data'], bins=bins, facecolor='none', edgecolor=pdfParams['color'], density=True)

        cdf = dist['distribution'].cdf(bins)
        cdf_ax.plot(bins, cdf, color=pdfParams['color'], label=pdfParams['label'])

        pdf_ax.set_title('Distribution density')
        cdf_ax.set_title('Cumulative density')
        pdf_ax.grid(lw=.5, alpha=.5, color='C7')
        cdf_ax.grid(lw=.5, alpha=.5, color='C7')
        pdf_ax.set_ylim(0,15)

        if dist.get('label') is not None:
            pdf_ax.legend(fontsize=9)
            cdf_ax.legend(fontsize=9)
    return fig