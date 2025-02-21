import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# given the x,y locations of a given node will return the velocity of that node
# at every timestep
def find_node_velocity(node_locs):
    vels = np.zeros((node_locs.shape[0] - 1))
    for j in range(node_locs.shape[0] - 1):
        vels[j] = np.sqrt(np.sum(np.square(node_locs[j] - node_locs[j+1])))
    # anything that nan is changed to 0 to make my life easier
    vels = [0 if np.isnan(x) else x for x in vels]
    return vels

def get_stats(node_vels):
    mean, std = np.mean(node_vels), np.std(node_vels)
    return mean, std, mean - (3 * std), mean + (3 * std)

def graph_vels(node_vels, check=False, old_low=None, old_high=None):
    if check:
        mean, std, low, high = get_stats(node_vels)
        print(mean, std, low, high)
        plt.figure(figsize=(15, 6))
        plt.plot(node_vels, '.-')
        if old_low != None and old_high != None:
            plt.hlines(old_high, 0, len(node_vels), color='r', linestyle='--')
            plt.hlines(old_low, 0, len(node_vels), color='r', linestyle='--')
        else:
            plt.hlines(high, 0, len(node_vels), color='r', linestyle='--')
            plt.hlines(low, 0, len(node_vels), color='r', linestyle='--')
        plt.show()

def nan_vals(locations):
    flat = locations.flatten()
    return 100 * (np.sum(np.isnan(flat)) / np.prod(locations.shape))

# FROM SLEEP ANALYSIS NOTEBOOK
def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))

        if x.shape[0] == 0 or y[x].shape[0] == 0:
            print('could not fill locations...')
            Y = Y.reshape(initial_shape)
            return Y
        # print(x.shape, y.[x]shape)
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y