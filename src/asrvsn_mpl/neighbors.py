import numpy as np
import matplotlib.pyplot as plt

ngon_colors = {
    3: 'brown',
    4: 'pink',
    5: 'blue',
    6: 'orange',
    7: 'green',
    8: 'purple',
    9: 'red',
    10: 'yellow',
    11: 'cyan',
    12: 'brown',
    13: 'pink',
}

def nb_hist(ax, nbs, show_nums: bool=False):
    support = np.arange(nbs.min(), nbs.max() + 1, dtype=int)
    idx_5gon = np.where(support == 5)[0][0]
    idx_7gon = np.where(support == 7)[0][0]
    colors = [ngon_colors[n] for n in support]
    freq = np.array([np.sum(nbs == n) for n in support])
    freq_5gon = freq[idx_5gon]
    freq_7gon = freq[idx_7gon]
    print(f'5-gon frequency: {freq_5gon}')
    print(f'7-gon frequency: {freq_7gon}')
    ax.bar(support, freq, color=colors)
    if show_nums:
        for s, f in zip(support, freq):
            ax.text(s, f+0.5, str(f), ha='center', va='bottom')
    ax.set_xlabel('Number of Neighbors')
    ax.set_ylabel('Frequency')
