from data.uci import retrieve_uci_data
from equations import calculate_linear_regression_parameters
from model.uciml import UciMLRepo
from plot import plot_points, plot_line, show_plot
from utils import skip


@skip
def run_problem1(*, dataset: UciMLRepo) -> None:
    data = dataset.data.features
    length = data['Length'].to_numpy()
    diameter = data['Diameter'].to_numpy()
    m, b = calculate_linear_regression_parameters(x_series=length, y_series=diameter)

    plot_points(
        x_series=length,
        y_series=diameter,
        x_axis_label='Length',
        y_axis_label='Diameter',
        point_size=3,
    )
    plot_line(
        slope=m,
        intercept=b,
        line_size=3,
    )
    show_plot(title='Length vs Diameter', should_show_grid=True)


def run() -> None:
    abalone = retrieve_uci_data(repo_name='abalone')
    run_problem1(dataset=abalone)


if __name__ == '__main__':
    run()
