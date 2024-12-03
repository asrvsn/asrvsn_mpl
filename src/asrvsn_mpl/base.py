import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.colors import LightSource
import numpy as np
from scipy.spatial import Delaunay, SphericalVoronoi, geometric_slerp
from typing import Tuple, Union, Optional, Callable, List
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.transforms as mtransforms
from matplotlib.legend import Legend 
from matplotlib.collections import PathCollection
from matplotlib.patches import Patch
from matplotlib.image import AxesImage

import os

from microseg.utils.colors import *
import matgeo.utils.poly as pu
from matgeo.ellipsoid import Ellipsoid
from matgeo.plane import Plane, PlanarPolygon
from matgeo.triangulation import Triangulation

''' General utils '''

def use_tex_font():
    # Use system tex for math rendering
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath, tikz}')
    # Use latex-like font for titles
    mpl.rcParams['mathtext.fontset'] = 'stix' 
    mpl.rcParams['font.family'] = 'STIXGeneral'

def label_subplots_mosaic(fig, axs: dict, loc=(4/72, -5/72), fontsize=14, bbox_kwargs=dict(), locs=dict(), unlabeled=set(), dark=set(), **kwargs):
    for label, ax in axs.items():
        if not label in unlabeled:
            # label physical distance to the left and up:
            loc = locs.get(label, loc)
            trans = mtransforms.ScaledTranslation(loc[0], loc[1], fig.dpi_scale_trans)
            fn = ax.text2D if isinstance(ax, Axes3D) else ax.text # 3d axes has different API
            bbox_color = 'none' if label in dark else 'white'
            tx_color = 'white' if label in dark else 'black'
            my_args = dict(
                bbox=dict(facecolor=bbox_color, edgecolor='none', pad=4.0, alpha=0.5) | bbox_kwargs, 
                # fontweight='bold',
                fontsize=fontsize,
                # fontfamily='sans-serif',
                color=tx_color,
                va='top',
                # usetex=False,
            ) | kwargs
            fn(0.0, 1.0, label, transform=ax.transAxes + trans, **my_args)
            ax.set_anchor('NW')

def set_color_cycle(colors: list):
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def rasterize_fig(fig):
    for ax in fig.get_axes():
        for child in ax.get_children():
            if isinstance(child, AxesImage):
                child.set_rasterized(True)
            elif isinstance(child, Patch) and child not in ax.spines.values():
                child.set_rasterized(True)
            elif isinstance(child, PathCollection) and len(child.get_offsets()) > 20:
                child.set_rasterized(True)
            else:
                child.set_rasterized(False)
    for spine in ax.spines.values():
        spine.set_rasterized(False)
    return fig


''' 2d plot utils '''

plots = []
gridspec = dict()
rel_figsize = (12,12)
ax_title_size = 20
ax_title_y = 0.98
tight_layout_rect = [0, 0.03, 1, 0.95]
def_mosaic_height = 4 # Per panel 
def_mosaic_width = 4 # Per panel
chars = 'abcdefghijklmnopqrstuvwxyz'
upperchars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def set_gridspec(gs: dict):
    global gridspec
    gridspec = gs

def set_figsize(size: Tuple[float, float]):
    '''
    Height x Width, like the rest of the world (but opposite to matplotlib).
    '''
    global rel_figsize
    rel_figsize = size

def set_layout_rect(rect: Tuple[float, float, float, float]):
    global tight_layout_rect
    tight_layout_rect = rect

def set_title_size(size: float):
    global ax_title_size
    ax_title_size = size

def plot_ax(index: Tuple[int, int], title: str, func, *args, **kwargs):
    global plots
    max_y = max(index[0]+1, len(plots))
    max_x = max(index[1]+1, max([0] + [len(row) for row in plots]))
    while len(plots) < max_y:
        plots.append([])
    for y in range(len(plots)):
        while len(plots[y]) < max_x:
            plots[y].append(None)
    plots[index[0]][index[1]] = (title, [func], [args], kwargs)

def modify_ax(index: Tuple[int, int], func, *args):
    global plots
    assert index[0] < len(plots), 'y axis out of range'
    assert index[1] < len(plots[index[0]]), 'x axis out of range'
    assert plots[index[0]][index[1]] != None, 'no existing plot to modify'
    plots[index[0]][index[1]][1].append(func)
    plots[index[0]][index[1]][2].append(args)

def render_plots():
    global plots, gridspec, rel_figsize
    assert all(len(plots[y]) == len(plots[0]) for y in range(len(plots))), 'plots array is ragged'
    if len(plots) > 0:
        nr, nc = len(plots), len(plots[0])
        fig = plt.figure(figsize=(rel_figsize[1]*nc, rel_figsize[0]*nr), constrained_layout=True)
        subfigs = fig.subfigures(nr, nc, **gridspec)
        for y, row in enumerate(plots):
            for x, element in enumerate(row):
                if element != None:
                    (title, funcs, argss, kwargss) = element
                    if nr*nc == 1:
                        sf = subfigs
                    elif nr == 1 or nc == 1:
                        sf = subfigs[max(y,x)]
                    else:
                        sf = subfigs[y,x]
                    ax = sf.add_subplot(**kwargss)
                    if not (title is None):
                        sf.suptitle(title, fontsize=ax_title_size, y=ax_title_y)
                        # ax.title.set_y(1.05)
                    # ax.figure.set_size_inches(6, 6)
                    for f, args in zip(funcs, argss):
                        f(ax, *args)
        plt.tight_layout(rect=tight_layout_rect)
        return True
    else:
        print('No axes found, not plotting.')

def show_plots(reset: bool=False):
    if render_plots():
        plt.show()
        plt.close()
        if reset:
            reset_plots()

def save_plots(path: str, separate: bool=False):
    global plots
    path = os.path.abspath(path)
    if separate:
        os.makedirs(path, exist_ok=True)
        for y, row in enumerate(plots):
            for x, element in enumerate(row):
                if element != None:
                    (title, funcs, argss, kwargss) = element
                    fig = plt.figure(frameon=False)
                    sf = fig.subfigures(1, 1)
                    ax = sf.add_subplot(**kwargss)
                    if not (title is None):
                        sf.suptitle(title, fontsize=ax_title_size, y=ax_title_y)
                    for f, args in zip(funcs, argss):
                        f(ax, *args)
                    plt.tight_layout()
                    plt.savefig(path + f'/{y}_{x}', bbox_inches='tight', pad_inches=0)
                    plt.close()
    else:
        if render_plots():
            bname = os.path.dirname(path)
            os.makedirs(bname, exist_ok=True)
            # matplotlib.use('Agg')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            print('Major tom to ground control, no plots to save.')

def reset_plots():
    global plots
    plots = []

def save_plot(path, funs):
    bname = os.path.dirname(path)
    os.makedirs(bname, exist_ok=True)
    fig, ax = plt.subplots(frameon=False)
    for fun in funs:
        fun(ax)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def clear_directory(path: str):
    if os.path.exists(path):
        assert os.path.isdir(path), 'path is not a directory'
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))

def convert_pngs_to_mp4(
        png_dir: str,
        fps: int=10,
    ): 
    '''
    Convert all pngs in a directory to an mp4
    '''
    assert os.path.isdir(png_dir), 'path is not a directory'
    pngs = sorted([img for img in os.listdir(png_dir) if img.endswith('.png')])
    assert len(pngs) > 0, 'no pngs found in directory'
    cmd = f'ffmpeg -r {fps} -f image2 -i {png_dir}/%d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -crf 25 -pix_fmt yuv420p -y {png_dir}/video.mp4'
    print('Running command:')
    print(cmd)
    os.system(cmd)
    # mp4_path = os.path.join(png_dir, 'video.mp4')
    # frame = cv2.imread(os.path.join(png_dir, pngs[0]))
    # height, width, layers = frame.shape
    # video = cv2.VideoWriter(mp4_path, 0, fps, (width,height))
    # for png in pngs:
    #     video.write(cv2.imread(os.path.join(png_dir, png)))
    # cv2.destroyAllWindows()
    # video.release()
    
def remove_spines(ax: plt.Axes, spines=None):
    if spines is None:
        spines = ['top', 'right', 'bottom', 'left']
    else:
        spines = [{'t': 'top', 'r': 'right', 'b': 'bottom', 'l': 'left'}[c] for c in spines]
    for sp in spines:
        ax.spines[sp].set_visible(False)

def ax_noframe(ax, hide_x: bool=False, hide_y: bool=False):
    for sp in ['top', 'bottom', 'left', 'right']:
        ax.spines[sp].set_visible(False)
    if hide_x:
        ax.get_xaxis().set_visible(False)
    if hide_y:
        ax.get_yaxis().set_visible(False)

def hide_axes(ax):
    remove_spines(ax)
    ax.set_yticks([])
    ax.set_xticks([])

def default_mosaic(mosaic: list):
    return plt.subplot_mosaic(mosaic, figsize=(def_mosaic_width * len(mosaic[0]), def_mosaic_height * len(mosaic)))

def mosaic_3d(layout: list, **kwargs):
    fig, axs = plt.subplot_mosaic(layout, **kwargs)
    nrows, ncols = len(layout), len(layout[0])
    for i in range(nrows):
        for j in range(ncols):
            axs[layout[i][j]].remove()
            axs[layout[i][j]] = plt.subplot(nrows, ncols, i * nrows + j + 1, projection='3d')
    return fig, axs

def replace_3d_mosaic(axs: dict, layout: list, key: Any):
    nrows, ncols = len(layout), len(layout[0])
    for i in range(nrows):
        for j in range(ncols):
            if layout[i][j] == key:
                axs[key].remove()
                axs[key] = plt.subplot(nrows, ncols, i * ncols + j + 1, projection='3d')
                return

''' Annotation '''

def label_ax(ax, i, **kwargs):
    title = i if type(i) is str else upperchars[i]
    kwargs = dict(xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=22) | kwargs
    ax.annotate(r'\textbf{'+title+r'} ', **kwargs)

def title_ax(ax, title: str, **kwargs):
    kwargs = dict(xy=(0.5, 1.05), xycoords='axes fraction', ha='center', va='center', fontsize=22) | kwargs
    ax.annotate(title, **kwargs)

def left_title_ax(ax, title: str, offset: float=-0.05, **kwargs):
    kwargs = dict(xy=(offset, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=22, rotation=90) | kwargs
    ax.annotate(title, **kwargs)


''' 1d/2d plots '''

def ax_im_gray(ax, data: np.ndarray, invert: bool=False):
    assert len(data.shape) == 2, 'data must be 2D'
    cmap = cm.gray_r if invert else cm.gray
    ax.imshow(data, cmap=cmap)
    ax.axis('off')

def ax_labels(ax, coords, labels, **kwargs):
    for (x, y), l in zip(coords, labels):
        ax.text(x, y, str(l), **kwargs)

def ax_arrows(ax, coords: np.ndarray, vectors: np.ndarray, max_len: float=1, **kwargs):
    lengths = np.linalg.norm(vectors, axis=1)
    vectors_ = max_len * vectors / lengths.max()
    for (x, y), (u, v) in zip(coords, vectors_):
        ax.arrow(x, y, u, v, length_includes_head=True, **kwargs)

def ax_1d(ax, interval: Tuple[int, int], xs: np.ndarray):
    '''
    Plot points on an interval with only the ends labeled
    '''
    ax.set_xlim(interval)
    ax.set_xticks([interval[0], interval[1]])
    ax.set_xticklabels([interval[0], interval[1]])
    ax.scatter(xs, np.zeros_like(xs), marker='o', color='black', s=10)
    # Draw a line at y=0
    ax.axhline(y=0, color='black', linewidth=1)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['bottom'].set_position(('data', 0))
    # gkde = scipy.stats.gaussian_kde(xs)
    # sxs = np.linspace(interval[0], interval[1], 1000)
    # samples = gkde.evaluate(sxs)
    # ax.plot(sxs, samples, linewidth=1)

def ax_periodic_y(ax, xs: np.ndarray, ys: np.ndarray, bounds: Tuple[float, float], **kwargs):
    assert bounds[1] > bounds[0], 'bounds must be increasing'
    assert xs.shape == ys.shape, 'xs and ys must have same shape'
    mask = np.logical_or(ys > bounds[1], ys < bounds[0])
    mask = np.insert(False, 0, np.abs(np.diff(mask)) == 1)
    ys = (ys - bounds[0]) % (bounds[1] - bounds[0]) + bounds[0]
    masked_ys = np.ma.MaskedArray(ys, mask)
    ax.plot(xs, masked_ys, **kwargs)

def ax_hist(ax, data: np.ndarray, bins: int=10, **kwargs):
    ax.hist(data, bins=bins, **kwargs)
    # ax_noframe(ax, hide_x=True)
    # ax.axis('off')

def ax_mean_bins(ax, xdata: np.ndarray, ydata: np.ndarray, bins: int=10, logscale: bool=False, stderr: bool=False, color='black', s=8):
    '''
    Plot the mean of ydata in bins of xdata 
    Reject data points using median absolute deviation threshold (set to inf to disable)
    '''
    if logscale:
        bins = np.logspace(np.log10(min(xdata)), np.log10(max(xdata)), bins)
    counts, bin_edges = np.histogram(xdata, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_avg = np.histogram(xdata, bins=bins, weights=ydata)[0] / counts
    # ax.bar(bin_centers, bin_avg, width=np.diff(bin_edges), align='center', alpha=0.7)
    ax.scatter(bin_centers, bin_avg, marker='o', color=color, s=s)
    if stderr:
        bin_stderr = np.sqrt(np.histogram(xdata, bins=bins, weights=ydata**2)[0] / counts - bin_avg**2) / np.sqrt(counts)
        ax.errorbar(bin_centers, bin_avg, yerr=bin_stderr, fmt='none', ecolor=color, capsize=2)
    if logscale:
        ax.set_xscale('log')

def ax_circle(ax, center: Tuple[float, float], radius: float, **kwargs):
    ts = np.linspace(0,1,100)
    x = radius * np.cos(2 * np.pi * ts) + center[0]
    y = radius * np.sin(2 * np.pi * ts) + center[1]
    # print('got here', center, radius)
    ax.plot(x,y, **kwargs)

def ax_heatmap(ax, h: np.ndarray):
    ax.imshow(h)
    ax.axis('off')

def ax_poly2d(ax: plt.Axes, vertices: np.ndarray, faces: list, facecolors=None, alpha: float=0.5):
    assert vertices.ndim == 2
    assert vertices.shape[1] == 2
    for i, f in enumerate(faces):
        poly = vertices[f].tolist()
        poly = np.array(poly + [poly[0]])
        ax.plot(poly[:,0], poly[:,1], color='black', linewidth=0.5)
        if not (facecolors is None):
            # xs = poly[:,0].tolist()
            # ys = poly[:,1].tolist()
            # ax.fill(xs + [xs[0]], ys + [ys[0]], [int(x*255) for x in facecolors[i]], alpha=alpha)
            # Draw filled polygon using patch
            polygon = mpatches.Polygon(poly, facecolor=facecolors[i], alpha=alpha)
            ax.add_patch(polygon)
    ax.set_aspect('equal')
    ax.set_axis_off()
    # ax.disconnect_zoom = zoom_factory(ax)

def ax_planar_polygons(ax: plt.Axes, polygons: List[PlanarPolygon], cmap: Callable=None, fillcolor=None, alpha: float=0.5, zorder: int=0, labels: bool=False, **kwargs):
    if cmap is None:
        if fillcolor is None:
            cmap = lambda polys: map_colors(np.arange(len(polygons)), 'categorical')
        else: 
            cmap = lambda _: [fillcolor] * len(polygons)
    colors = cmap(polygons)
    assert len(colors) == len(polygons)
    patchkw = {'alpha': alpha, 'zorder': zorder} | ({'transform': kwargs['transform']} if 'transform' in kwargs else {})
    for i, poly in enumerate(polygons):
        p = np.array(poly.to_shapely().exterior.coords)
        patch = mpatches.Polygon(p, facecolor=colors[i], **patchkw)
        ax.add_patch(patch)
        if kwargs.get('linewidth', 1) > 0:
            pkwargs = kwargs.copy()
            if 'linecolor' in pkwargs:
                pkwargs['color'] = pkwargs['linecolor']
                del pkwargs['linecolor']
            ax.plot(p[:,0], p[:,1], **pkwargs)
        if labels:
            # Add label "i"
            mu = poly.centroid()
            ax.text(mu[0], mu[1], str(i), fontsize=8, color='white')
    ax.set_aspect('equal')
    # ax.set_axis_off()

def ax_ppoly(ax: plt.Axes, poly: PlanarPolygon, color='grey', **kwargs):
    ax_planar_polygons(ax, [poly], cmap=lambda _: [color], **kwargs)

def ax_lines(ax: plt.Axes, x1: np.ndarray, x2: np.ndarray, **kwargs):
    assert x1.shape == x2.shape
    assert x1.ndim == 2
    assert x1.shape[1] == 2
    for i in range(x1.shape[0]):
        ax.plot([x1[i,0], x2[i,0]], [x1[i,1], x2[i,1]], **kwargs)

def ax_rect(ax: plt.Axes, center: np.ndarray, wh: Tuple[float, float], fill_color: str=None, **kwargs):
    w, h = wh
    assert w > 0
    assert h > 0
    xs = [center[0] - w, center[0] + w, center[0] + w, center[0] - w, center[0] - w]
    ys = [center[1] - h, center[1] - h, center[1] + h, center[1] + h, center[1] - h]
    if fill_color is not None:
        ax.fill(xs, ys, fill_color)
    ax.plot(xs, ys, **kwargs)

def ax_square(ax: plt.Axes, center: np.ndarray, r: float, **kwargs):
    ax_rect(ax, center, (r, r), **kwargs)

def ax_inset(ax: plt.Axes, center: np.ndarray, wh: Tuple[float, float], zorder: int=2, fill_color: str='white', **kwargs) -> plt.Axes:
    # Add inset
    x, y = center
    w, h = wh
    inset_rect = PlanarPolygon.from_center_wh(center, wh)
    ax_planar_polygons(ax, [inset_rect], cmap=lambda _: [fill_color], alpha=1, zorder=zorder, **kwargs)
    iax = ax.inset_axes([x-w, y-h, w*2, h*2], transform=ax.transData, zorder=zorder)
    return iax

def ax_rect_callout(
        ax: plt.Axes,
        center1: np.ndarray,
        wh1: Tuple[float, float],
        center2: np.ndarray,
        wh2: Tuple[float, float],
        zorder: int=2,
        fill_color: str='white',
        **kwargs
    ) -> plt.Axes:
    w1, h1 = wh1
    w2, h2 = wh2
    assert w1 > 0
    assert h1 > 0
    assert w2 > 0
    assert h2 > 0
    ax_rect(ax, center1, wh1, **kwargs)
    # ax_rect(ax, center2, wh2, **kwargs)
    if center1[0] < center2[0]:
        x1_off = w1
        x2_off = -w2
    else:
        x1_off = -w1
        x2_off = w2
    ax.plot([center1[0] + x1_off, center2[0] + x2_off], [center1[1]+h1, center2[1]+h2], **kwargs)
    ax.plot([center1[0] + x1_off, center2[0] + x2_off], [center1[1]-h1, center2[1]-h2], **kwargs)
    iax = ax_inset(ax, center2, wh2, zorder=zorder, fill_color=fill_color, **kwargs)
    return iax

def ax_square_callout(ax: plt.Axes, center1: np.ndarray, r1: float, center2: np.ndarray, r2: float, **kwargs):
    ax_rect_callout(ax, center1, (r1, r1), center2, (r2, r2), **kwargs)

def ax_scale_bar(ax, length, upp: float=1.0, units='', right_pad=0.05, bottom_pad=0.05, text_pad=0.07, label=True, fontsize=16, color='white', **kwargs):
    '''
    upp: units per pixel (e.g. microns)
    '''
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pad_data = right_pad * (xlim[1] - xlim[0])
    y_pad_data = bottom_pad * (ylim[1] - ylim[0])
    pix_length = length / upp
    x_start = xlim[1] - x_pad_data - pix_length
    x_end = xlim[1] - x_pad_data
    y_pos = ylim[0] + y_pad_data
    kwargs = dict(color=color, linewidth=4) | kwargs
    ax.plot([x_start, x_end], [y_pos, y_pos], **kwargs)
    if label:
        # y_pos += text_pad * (ylim[1] - ylim[0])
        # ax.text((x_start + x_end) / 2, y_pos, f'{length:.0f} {units}', ha='center', va='bottom', fontsize=fontsize, color=color, bbox=dict(facecolor='none', edgecolor=color, margin=8.0))
        x_text = 1 - right_pad - pix_length/(2*(xlim[1]-xlim[0]))
        ax.annotate(f'{length:.0f} {units}', xy=(x_text, text_pad), xycoords='axes fraction', ha='center', va='bottom', fontsize=fontsize, color=color)

def ax_orientations(ax: plt.Axes, centers: np.ndarray, polygons: List[PlanarPolygon], colors=['grey', 'white'], **kwargs):
    ''' Plot orientation crossbar of polygons using their second moments ''' 
    assert centers.ndim == 2
    assert centers.shape[1] == 2
    assert len(polygons) == centers.shape[0]
    for c, poly in zip(centers, polygons):
        vs = poly.principal_axes()
        rs = poly.elliptic_radii()
        for v, r, co in zip(vs, rs, colors):
            e1 = c - r * v
            e2 = c + r * v
            ax.plot([e1[0], e2[0]], [e1[1], e2[1]], color=co, **kwargs)

def ax_ellipse(ax: plt.Axes, ellipse: Ellipsoid, axes: bool=False, major_axis_color='blue', minor_axis_color='red', axes_linewidth=1, fill_color=None, fill_alpha=1., **kwargs):
    assert ellipse.ndim == 2
    theta = np.linspace(0, 2*np.pi, 100)
    circ = np.array([np.cos(theta), np.sin(theta)]).T
    pts = ellipse.map_sphere(circ)
    ax.plot(pts[:,0], pts[:,1], **kwargs)
    exargs = dict(transform=kwargs['transform']) if 'transform' in kwargs else {}
    if axes:
        # Draw major & minor axes
        P, rs = ellipse.get_axes_stretches()
        for i in range(ellipse.ndim):
            e1 = ellipse.v + P[:,i] * rs[i]
            e2 = ellipse.v - P[:,i] * rs[i]
            ax.plot([e1[0], e2[0]], [e1[1], e2[1]], color=[major_axis_color, minor_axis_color][i], linewidth=axes_linewidth, **exargs)
    if not fill_color is None:
        ax.fill(pts[:,0], pts[:,1], facecolor=fill_color, alpha=fill_alpha, **exargs)

def ax_ellipses(ax: plt.Axes, ellipses: List[Ellipsoid], **kwargs):
    for e in ellipses:
        ax_ellipse(ax, e, **kwargs)

def ax_tri_2d(ax: plt.Axes, pts: np.ndarray, simplices: np.ndarray, **kwargs):
    assert pts.shape[1] == 2
    ax_poly2d(ax, pts, simplices, facecolors=['blue'] * simplices.shape[0], **kwargs)
    ax.set_axis_off()
    ax.set_aspect('equal')

''' Legend tools '''

def ax_color_legend(ax, labels, colors, loc='upper right', horizontal=False, alpha=1., **kwargs):
    ax.legend(handles=[
        mpatches.Patch(color=c, label=str(l), alpha=alpha, **kwargs)
        for l, c in zip(labels, colors)
    ], loc=loc, ncol=(len(labels) if horizontal else 1), columnspacing=0.5, handlelength=0.7, handletextpad=0.4)

def ax_categorical_legend(ax, data: np.ndarray, **kwargs):
    labels = np.unique(data)
    colors = map_colors(labels, 'categorical')
    ax_color_legend(ax, labels, colors, **kwargs)

def ax_polar_legend(ax, speed: float=1):
    '''
    Plot polar legend with time multiplier speed
    '''
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx, sy = np.sign(x1 - x0), np.sign(y1 - y0)
    hw = min(sx*(x1 - x0), sy*(y1 - y0)) / 8
    r = hw/2
    margin = r/3
    n = 100
    xs, ys = np.linspace(-r, r, n), np.linspace(-r, r, n)
    padding = r + margin
    ox, oy = x1 - sx * padding, y1 - sy * padding
    zs = (np.arctan2.outer(ys, xs) * speed) % (2*np.pi) 
    colors = map_colors(zs, 'periodic', d_min=0, d_max=2*np.pi)
    ax.pcolormesh(xs + ox, ys + oy, colors)

def ax_deduped_legend(ax, **kwargs):
    # Deduplicates & orders labels alphabetically
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(sorted(zip(labels, handles), key=lambda t: t[0]))
    lw = None
    if 'linewidth' in kwargs:
        lw = kwargs.pop('linewidth')
    leg = ax.legend(by_label.values(), by_label.keys(), **kwargs)
    if lw is not None:
        for lh in leg.get_lines(): 
            lh.set_linewidth(lw)

''' 3d plots '''

def ax_im_gray_3d(ax, data: np.ndarray, height: np.ndarray, invert: bool=False):
    assert len(data.shape) == 2, 'data must be 2D'
    assert data.shape == height.shape, 'data and height must have same shape'
    cmap = cm.gray_r if invert else cm.gray
    # Create x, y meshgrid
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    # Plot
    ax.scatter(xx, yy, height[xx,yy], c=data[xx,yy], cmap=cmap)
    # ax.scatter(data[:,0], data[:,1], height, c=data, cmap=cmap)
    ax.set_aspect('equal')
    ax.set_axis_off()

def ax3d_equalize(ax):
    ''' https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def ax_heatmap_3d(ax, h: np.ndarray):
    x = np.arange(h.shape[0])
    y = np.arange(h.shape[1])
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, h, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_axis_off()
    ax.set_aspect('equal')
    # ax3d_equalize(ax)
    # ax.disconnect_zoom = zoom_factory(ax)

def ax_poly(ax: plt.Axes, vertices: np.ndarray, faces: list, **kwargs):
    ''' Plot a polyhedron '''
    polygons = [vertices[f] for f in faces]
    ax_poly_raw(ax, polygons, **kwargs)

def ax_poly_raw(ax: plt.Axes, polygons: List[np.ndarray], scatter: bool = False, facecolors=None, edgecolor: str='black', lim=None, mult=1., altaz=(60, 0), scatter_kwargs=dict(), **kwargs):
    ''' Plot a polyhedron '''
    alt, az = altaz
    vertices = np.concatenate(polygons)
    assert vertices.shape[1] == 3
    if facecolors is None:
        color = ax._get_lines.get_next_color()
        facecolors = np.array(mcolors.to_rgba(color))
    art_kwargs = dict(
        edgecolor='black',
        facecolors=facecolors,
        alpha=1.,
        shade=True,
        lightsource=LightSource(altdeg=alt, azdeg=az), # Ensure the light shines
        zorder=0, 
        linewidth=0.5,
    ) | kwargs
    pc = art3d.Poly3DCollection(polygons, **art_kwargs)
    ax.add_collection(pc)
    if scatter:
        scatter_kwargs = dict(color='lightgreen', s=2) | scatter_kwargs
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], **scatter_kwargs)
    if lim is None:
        ax.auto_scale_xyz(vertices[:,0]*mult, vertices[:,1]*mult, vertices[:,2]*mult) # poly3d doesn't do this
    else:
        assert len(lim) == 3
        # pdb.set_trace()
        ax.set_xlim(-lim[0], lim[0])
        ax.set_ylim(-lim[1], lim[1])
        ax.set_zlim(-lim[2], lim[2])
    ax.set_aspect('equal')
    # Remove axis
    ax.axis('off')
    # ax.disconnect_zoom = zoom_factory(ax)
    # Set view in alt/az coordinates
    ax.view_init(alt, az)

def ax_poly_normals(ax: plt.Axes, vertices: np.ndarray, faces: list, length: float=0.4):
    ''' Plot unit normal vectors to faces '''
    barycenters = np.array([vertices[f].mean(axis=0) for f in faces])
    normals = pu.face_normals(vertices, faces)
    assert barycenters.shape == normals.shape
    ax.quiver(barycenters[:, 0], barycenters[:, 1], barycenters[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], length=length, color='red', zorder=100)

def ax_faces(ax: plt.Axes, faces: list):
    pc = art3d.Poly3DCollection(faces, edgecolor='black', linewidth=0.5, zorder=0)
    ax.add_collection(pc)
    lim = max(np.max(np.abs(f)) for f in faces) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_aspect('equal')
    # Remove axis
    ax.set_axis_off()

def ax_tri(ax: plt.Axes, pts: np.ndarray, tri: np.ndarray, scatter=False, **kwargs):
    assert pts.shape[1] == 3
    # Set face color invisible
    ax.plot_trisurf(pts[:,0], pts[:,1], pts[:,2], triangles=tri, edgecolor='black', linewidth=0.5, zorder=0, **kwargs)
    # ax3d_equalize(ax)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if scatter:
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='lightgreen', s=2)
    # ax.disconnect_zoom = zoom_factory(ax)

def ax_triangulation(ax: plt.Axes, tri: Triangulation, **kwargs):
    ax_tri(ax, tri.pts, tri.simplices, **kwargs)

def ax_tri_wireframe(ax, points, tri):
    for tr in tri:
        pts = points[tr, :]
        ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
        ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
        ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
        ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
        ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
        ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')

    ax.scatter(points[:,0], points[:,1], points[:,2], color='b')
    ax3d_equalize(ax)
    ax.set_axis_off()

def ax_sphere(ax: plt.Axes, r:float=1.0, **kwargs):
    ''' Display the unit sphere '''
    kwargs = dict({'color': 'lightgreen', 'alpha': 0.2, 'shade': True}, **kwargs)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v)) * r
    y = np.outer(np.sin(u), np.sin(v)) * r
    z = np.outer(np.ones(np.size(u)), np.cos(v)) * r
    ax.plot_surface(x, y, z, **kwargs)
    r /= 1.6 # wtf
    ax.set_xlim(-r*1.01, r*1.01)
    ax.set_ylim(-r*1.01, r*1.01)
    ax.set_zlim(-r*1.01, r*1.01)
    ax.set_aspect('equal')

def ax_spherical_voronoi(ax: plt.Axes, sv: SphericalVoronoi, lines=True, points=True, s=2, c='darkgreen', colors: List[str]=None):
    ''' 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.SphericalVoronoi.html
    '''
    sv.sort_vertices_of_regions()
    ts = np.linspace(0, 1, 5)
    if points:
        ax.scatter(sv.points[:, 0], sv.points[:, 1], sv.points[:, 2], c=c, s=s)
    if lines:
        for region in sv.regions:
            n = len(region)
            for i in range(n):
                start = sv.vertices[region][i]
                end = sv.vertices[region][(i + 1) % n]
                result = geometric_slerp(start, end, ts)
                ax.plot(result[..., 0], result[..., 1], result[..., 2], color=c, linewidth=0.5, zorder=1)
    ax.set_axis_off()

def ax_scatter_sphere(ax: plt.Axes, pts: np.ndarray):
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    ax_sphere(ax)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='darkgreen', s=2)
    ax.set_axis_off()

def ax_ellipsoid(ax: plt.Axes, ell: Ellipsoid, color='y', alpha=0.3, **kwargs):
    X = ell.sample_mgrid()
    ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], color=color, alpha=alpha, **kwargs)
    ax.set_aspect('equal')

def ax_plane(ax: plt.Axes, pln: Plane, color='y', alpha=0.3, arrow=True, **kwargs):
    ''' Visualize a plane using offset-distance '''
    X = pln.sample_mgrid()
    ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], color=color, alpha=alpha, **kwargs)
    if arrow:
        nb = pln.b * pln.n
        ax.quiver(0, 0, 0, nb[0], nb[1], nb[2], color='r', zorder=100)
    ax.set_aspect('equal')

if __name__ == '__main__':
    # Test things
    plot_ax((0,0), None, lambda ax: ax.plot(np.arange(10), np.arange(10)))
    plot_ax((0,1), None, lambda ax: ax.plot(np.arange(10), np.arange(10)))
    modify_ax((0,1), lambda ax: ax_polar_legend(ax))
    show_plots()