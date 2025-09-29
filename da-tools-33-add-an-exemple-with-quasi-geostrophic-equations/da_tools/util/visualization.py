import torch
from matplotlib import pyplot as plt


def show_1dseqimg(
    x, dt=1.0, y_axis=None, x_label=None, ax=None, cmap=None, title=None, vmin=None, vmax=None, center_on_zero=False
):
    """Plots the evaluation of a multivariate time series through time.

    Args:
        x (torch.Tensor): the time series to represent
        dt: the time step in the time series
        y_axis: the y axis in the visualization
        x_label: the label to use for the x axis
        ax: the ax on which to display the visualization
        cmap (str): the colormap
        title (str): the title for the visualization
        vmin: the minimum value for the color range
        vmax: the maximum value for the color range
        center_y (bool): if true, automatically sets vmin and vmax in order to center the color range on zero
    """
    y_axis = torch.arange(0, x.shape[0]) if y_axis is None else y_axis
    if center_on_zero:
        vmax_abs = torch.max(torch.abs(x))
        vmin, vmax = -vmax_abs, vmax_abs
    cmap = "viridis" if cmap is None else cmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 3))
    time_axis = torch.arange(x.shape[1]) * dt
    img = ax.pcolor(time_axis, y_axis, x, cmap=cmap, vmin=vmin, vmax=vmax)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if title is not None:
        ax.set_title(title)
    plt.colorbar(img, ax=ax)
