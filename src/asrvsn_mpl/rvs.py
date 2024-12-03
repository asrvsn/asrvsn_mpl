'''
Random variable functions
'''

from typing import Dict, List, Tuple, Union, Optional
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def make_rv_title(rv: stats.rv_continuous, prms: tuple) -> str:
    '''
    Make a title for a random variable
    - X: exponential
    - Y: gamma
    - Z: beta
    - W: gaussian
    '''
    def add_scale(title: str, scale: float, scale_precision: int=3) -> str:
        if np.isclose(scale, 1, atol=10**(-scale_precision)):
            return title
        else:
            return f'{round(scale, scale_precision)}{title}'
    def add_loc(title: str, loc: float, loc_precision: int=3) -> str:
        if np.isclose(loc, 0, atol=10**(-loc_precision)):
            return title
        else:
            s = '+' if loc > 0 else '-'
            return f'{title} {s} {np.abs(round(loc, loc_precision))}'
    if rv == stats.expon:
        loc, scale = prms
        return add_loc(add_scale(f'$X_1$', scale), loc)
    elif rv == stats.gamma:
        k, loc, scale = prms
        return add_loc(add_scale(f'$Y_{{{round(k, 3)},1}}$', scale), loc)
    elif rv == stats.beta:
        a, b, loc, scale = prms
        return add_loc(add_scale(f'$Z_{{{round(a, 3)},{round(b, 3)}}}$', scale), loc)
    elif rv == stats.lognorm:
        s, loc, scale = prms
        return add_loc(add_scale(f'$\exp(W_{{0,{round(s, 3)}}})$', scale), loc)
    elif rv == stats.gengamma:
        a, k, loc, scale = prms
        return add_loc(add_scale(f'$Y\'_{{{round(a, 3)},{round(k, 3)}}}$', scale), loc)
    elif rv == stats.weibull_min:
        c, loc, scale = prms
        return add_loc(add_scale(f'$WB_{{{round(c, 3)},1}}$', scale), loc)
    elif rv == stats.betaprime:
        a, b, loc, scale = prms
        return add_loc(add_scale(f'$Z\'_{{{round(a, 3)},{round(b, 3)}}}$', scale), loc)
    elif rv == stats.invweibull: # Frechet random variable
        c, loc, scale = prms
        return add_loc(add_scale(f'$F_{{{round(c, 3)},1}}$', scale), loc)
    elif rv == stats.norm:
        loc, scale = prms
        return r'$'+f'W_{{{round(loc, 3)},{round(scale, 3)}}}'+r'$'
    elif rv == stats.invgamma:
        a, loc, scale = prms
        return add_loc(add_scale(f'$Y^{{-1}}_{{{round(a, 3)},1}}$', scale), loc)
    else:
        raise ValueError(f'Unknown random variable: {rv}')

def prms_to_dict(rv: stats.rv_continuous, prms: tuple) -> Dict[str, float]:
    '''
    Convert random variable parameters to dictionary; naming in scipy convention
    '''
    if rv == stats.expon:
        loc, scale = prms
        return {'loc': loc, 'scale': scale}
    elif rv == stats.gamma:
        a, loc, scale = prms
        return {'a': a, 'loc': loc, 'scale': scale}
    elif rv == stats.beta:
        a, b, loc, scale = prms
        return {'a': a, 'b': b, 'loc': loc, 'scale': scale}
    elif rv == stats.lognorm:
        s, loc, scale = prms
        return {'s': s, 'loc': loc, 'scale': scale}
    elif rv == stats.gengamma:
        a, c, loc, scale = prms
        return {'a': a, 'c': c, 'loc': loc, 'scale': scale}
    elif rv == stats.weibull_min:
        c, loc, scale = prms
        return {'c': c, 'loc': loc, 'scale': scale}
    elif rv == stats.betaprime:
        a, b, loc, scale = prms
        return {'a': a, 'b': b, 'loc': loc, 'scale': scale}
    elif rv == stats.invweibull: # Frechet random variable
        c, loc, scale = prms
        return {'c': c, 'loc': loc, 'scale': scale}
    elif rv == stats.norm:
        loc, scale = prms
        return {'loc': loc, 'scale': scale}
    else:
        raise ValueError(f'Unknown random variable: {rv}')

def fit_rv(data: np.ndarray, rv: stats.rv_continuous, loc: float=None, scale: float=None, pvalue: bool=False, pvalue_statistic: str='ks') -> Tuple[tuple, Optional[float]]:
    '''
    Fit rv. If pvalue is requested, return type is ((...params), pvalue) else ((...params), None)
    '''
    known_params = dict()
    if loc != None:
        known_params['loc'] = loc
    if scale != None:
        known_params['scale'] = scale
    fixed_params = {f'f{k}': v for k, v in known_params.items()}
    prms = rv.fit(data, **fixed_params)
    if pvalue:
        if pvalue_statistic == 'ks':
            rvi = rv(*prms) # Random variable instance  
            # Use exact calculation for Kolmogorov-Smirnov
            p = stats.ks_1samp(data, rvi.cdf).pvalue
        else:
            # Use Monte Carlo approximation for other statistics
            fit_params = {k: v for k, v in prms_to_dict(rv, prms).items() if not (k in known_params)}
            gof_res = stats.goodness_of_fit(rv, data, known_params=known_params, fit_params=fit_params, statistic=pvalue_statistic)
            p = gof_res.pvalue
    else:
        p = None
    return prms, p

def plot_hist_rvs(
            ax: plt.Axes, 
            data: np.ndarray, 
            rvs: List[stats.rv_continuous]=[], 
            loc: float=None, 
            scale: float=None, 
            xmin=None,
            xmax=None, 
            filter_xmin: bool=False,
            filter_xmax: bool=False,
            log: bool=True, 
            bins: int=50, 
            pvalues: bool=False, 
            pvalue_statistic: str='ks',
            legend_loc: str='upper right',
            linewidth: float=0.5, 
            alpha=0.5,
            fs: int=14,
            **kwargs
    ):
    '''
    Plot histogram of data with fitted random variables
    '''
    assert (scale is None) or scale > 0, 'Scale must be positive'
    if xmin is None:
        xmin = data.min()
    else:
        if filter_xmin:
            data = data[data >= xmin]
        else:
            assert data.min() >= xmin, 'xmin must be less than or equal to data.min()'
    if xmax is None:
        xmax = data.max()
    else:
        if filter_xmax:
            data = data[data <= xmax]
        else:
            assert data.max() <= xmax, 'xmax must be greater than or equal to data.max()'
    ax.hist(data, bins=bins, density=True, histtype='stepfilled', alpha=alpha, **kwargs)
    support = np.linspace(xmin, xmax, 100)
    for i, rv in enumerate(rvs):
        prms, p = fit_rv(data, rv, loc=loc, scale=scale, pvalue=pvalues, pvalue_statistic=pvalue_statistic)
        rvi = rv(*prms) # Random variable instance
        pdf = rvi.pdf(support)
        title = make_rv_title(rv, prms)
        if pvalues:
            ptitle = r'$p_{\text{' + pvalue_statistic + r'}}$'
            title += f' ({ptitle}={round(p, 3)})'
        linestyle = (0, (4*(i+1), 2))
        ax.plot(support, pdf, color='black', linestyle=linestyle, label=title, linewidth=linewidth)
    if len(rvs) > 0:
        ax.legend(loc=legend_loc, fontsize=fs)
    ax.set_xlim(xmin, xmax)
    if log:
        ax.set_yscale('log')

def plot_violin_rvs(
        ax: plt.Axes,
        all_data: Dict[str, np.ndarray], 
        rvs=[],
        loc: float=None,
        limits=(-np.inf, np.inf),  
        linewidth: float=0.5,
        color: str='black',
        s: float=1,
        marker: str='.',
        bins: int=50,
        means: bool=False,
        mean_color: str='black',
        hscale: float=0.8, # Hist scaling
        colors: Dict[str, str]={},
        orientation: str='xy',
        doublesided: bool=False,
        **kwargs
    ):
    '''
    Make violin-scatterplot using specified rvs.
    '''
    assert 0 < hscale <= 1, 'hscale must be in (0, 1]'
    assert orientation in ['xy', 'yx'], 'orientation must be xy or yx'
    if orientation == 'xy':
        m_ticks = ax.set_yticks
        m_ticklabels = ax.set_yticklabels
        m_lim = ax.set_ylim
        m_scatter = ax.scatter
        m_plot = ax.plot
        m_stairs = ax.stairs
    else:
        m_ticks = ax.set_xticks
        m_ticklabels = ax.set_xticklabels
        m_lim = ax.set_xlim
        m_scatter = lambda xs, ys, *args, **kwargs: ax.scatter(ys, xs, *args, **kwargs)
        m_plot = lambda xs, ys, *args, **kwargs: ax.plot(ys, xs, *args, **kwargs)
        m_stairs = lambda *args, **kwargs: ax.stairs(*args, **kwargs, orientation='horizontal')
    
    m_ticks(np.arange(len(all_data)))
    m_ticklabels(list(all_data.keys()))
    m_lim(-0.2, len(all_data))

    for pos, (key, data) in enumerate(all_data.items()):
        data = data[(data >= limits[0]) & (data <= limits[1])]
        m_scatter(data, np.full(data.shape, pos), color=color, s=s, marker=marker, **kwargs)
        hist, bin_edges = np.histogram(data, bins=50, density=True)
        scaling = hscale / hist.max() # Scale to fit between successive rows
        if doublesided:
            scaling *= 0.5
        hist *= scaling
        staircolor = colors.get(key, 'blue')
        m_stairs(hist + pos, bin_edges, color=staircolor, alpha=0.5, baseline=pos, fill=True)
        if doublesided:
            m_stairs(-hist + pos, bin_edges, color=staircolor, alpha=0.5, baseline=pos, fill=True)
        support = np.linspace(data.min(), data.max(), 100)
        for i, rv in enumerate(rvs):
            fit_args = dict()
            if loc != None:
                fit_args['floc'] = loc
            prms = rv.fit(data, **fit_args)
            rvi = rv(*prms)
            pdf = rvi.pdf(support)
            pdf *= scaling
            title = make_rv_title(rv, prms)
            linestyle = (0, (4*(i+1), 2))
            # m_plot(support, pdf + pos, color='black', linestyle=linestyle, linewidth=linewidth)
            # Add label at mode
            # ax.text(rvi.ppf(0.5), pdf.max() + pos, title, fontsize=10, horizontalalignment='center', verticalalignment='center')
        if means:
            # Draw vertical line at mean from pos to pos+1
            mean = data.mean()
            y_off = -hscale if doublesided else 0
            m_plot([mean, mean], [pos+y_off, pos+hscale], color=mean_color, linewidth=linewidth)
    