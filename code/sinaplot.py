import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde


def find_closest(A, target):
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


def sinaplot(y, pos, ax=None, direction='left', colour='grey', mean_colour='black',
             raw_alpha=0.8, raw_size=4,
             mean_size=30,
             kde_res=1000, kde_width=1, kde_alpha=0.25,
             cov_factor=0.25, x_dodge=0.01):
    """[summary]

    Args:
        y (numpy.ndarray): A single row of data
        pos (int): Position on the X axis
        direction (str, optional): Direction of the 'cloud'. Defaults to 'left'.
        colour (str, optional): Colour of the raw data and cloud. Defaults to 'grey'.
        mean_colour (str, optional): Colour of the mean and CI. Defaults to 'black'.
        raw_alpha (float, optional): Alpha for raw scatter. Defaults to 0.8.
        raw_size (int, optional): Size of raw scatter. Defaults to 4.
        mean_size (int, optional): Size of mean scatter. Defaults to 30.
        kde_res (int, optional): Number of bins in KDE. Defaults to 1000.
        kde_width (int, optional): X axis width of KDE. Defaults to 1.
        kde_alpha (float, optional): 'Cloud' alpha. Defaults to 0.25.
        cov_factor (float, optional): Smoothness of KDE. Defaults to 0.25.
        x_dodge (float, optional): Distance between mean and pos. Defaults to 0.01.
    """
    if ax is None:
        ax = plt.gca()

    # kde plot
    prob_density = kde.gaussian_kde(y)
    prob_density.covariance_factor = lambda: cov_factor
    prob_density._compute_covariance()
    kde_y = np.linspace(np.min(y), np.max(y), kde_res)

    if direction == 'left':
        kde_x = prob_density(kde_y) * -1 * kde_width + pos
        mean_pos = pos - x_dodge
    elif direction == 'right':
        kde_x = prob_density(kde_y) * kde_width + pos
        mean_pos = pos + x_dodge

    kde_x2 = np.ones((kde_x.shape)) * pos

    # draw kde violin type plot
    ax.fill_betweenx(y=kde_y, x1=kde_x, x2=kde_x2,
                      interpolate=True, alpha=kde_alpha,
                      color=colour, linewidth=0)

    # draw the raw data with kde restrictions
    x = np.zeros((len(y)))
    for i in range(len(y)):
        idx = find_closest(kde_y, y[i])
        jitter = np.random.uniform(0, kde_x[idx]-pos)
        x[i] = pos+jitter

    ax.scatter(x, y, color=colour, alpha=raw_alpha,
                s=raw_size, linewidth=0, zorder=0)

    # draw cis
    # Confidence Interval = x̄ ± z * ơ / √n
    ci_lo = np.mean(y) - 1.96 * np.std(y) / np.sqrt(len(y))
    ci_hi = np.mean(y) + 1.96 * np.std(y) / np.sqrt(len(y))
    ax.plot([mean_pos, mean_pos], [ci_lo, ci_hi], lw=1, color=mean_colour)

    # draw means
    ax.scatter(mean_pos, np.mean(y), alpha=1, s=mean_size,
                linewidth=0, color=mean_colour)
    return ax
