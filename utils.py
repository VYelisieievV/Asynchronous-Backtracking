import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_chess(field):
    """Plot chessboard like figure.

    Args:
        field (np.ndarray(dtype=np.uint8)): Array with figures coloured on it.
    """
    plt.figure()
    im = plt.imshow(field)

    ax = plt.gca()
    n = field.shape[0]
    # Major ticks
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, n + 1, 1))
    ax.set_yticklabels(np.arange(1, n + 1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)


def nogood_isin_nogoods(nogood, nogoods):
    """Utility for checking if list with np.ndarray in it is inside another list.

    Args:
        nogood (list): nogood, list of pairs (agent, value)
        nogoods (list): list of noogoods

    Returns:
        bool: True if nogood in nogoods
    """
    for el in nogood:
        el[1] = tuple(el[1])
    for ng in nogoods:
        for el in ng:
            el[1] = tuple(el[1])

    isin = nogood not in nogoods
    for el in nogood:
        el[1] = np.array(el[1])
    for ng in nogoods:
        for el in ng:
            el[1] = np.array(el[1])

    return isin


def normalize_nogood(nogood_list):
    """Drop duplicates from nogood. Maybe possible to drop pandas dependency.

    Args:
        nogood_list (list): nogood, list of pairs (agent, value)

    Returns:
        [list]: filtered list
    """
    df = pd.DataFrame(nogood_list)
    return df[~df[1].apply(tuple).duplicated()].values.tolist()
