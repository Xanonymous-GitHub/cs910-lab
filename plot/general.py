from collections.abc import Sequence
from numbers import Real

from matplotlib import pyplot as plt
from numpy import ndarray, array, float64


def plot_points(
        *,
        x_series: ndarray[Real] | Sequence[Real],
        y_series: ndarray[Real] | Sequence[Real],
        x_axis_label: str,
        y_axis_label: str,
        point_size: int = 5,
) -> None:
    # plot points
    plt.plot(x_series, y_series, '.', markersize=point_size)

    # add labels
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)


def plot_line(
        *,
        slope: float64,
        intercept: float64,
        show_equation: bool = True,
        line_size: int = 5,
) -> None:
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    label = f'y = {slope:.4f}x {'+' if intercept > 0 else '-'} {abs(intercept):.4f}' if show_equation else None
    plt.plot(x_vals, y_vals, '--', markersize=line_size, label=label)


def show_plot(*, title: str, should_show_grid: bool = False) -> None:
    # add title
    plt.title(title)
    manager = plt.get_current_fig_manager()
    manager.set_window_title(title)

    # enable grid
    plt.grid(should_show_grid)

    # show legend
    plt.legend(loc='best')

    # show plot
    plt.show()

    # clear plot
    plt.clf()
    plt.cla()
    plt.close('all')
