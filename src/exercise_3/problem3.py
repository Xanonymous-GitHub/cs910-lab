from data.uci import retrieve_uci_data
from model.uciml import UciMLRepo
from plot import plot_points, show_plot
from plot.stats import create_regression_stat_model


def run_problem3(*, dataset: UciMLRepo) -> None:
    data = dataset.data.original
    whole_weight = data['Whole_weight'].to_numpy()
    diameter = data['Diameter'].to_numpy()

    model_a = create_regression_stat_model(
        data,
        'Whole_weight ~ Diameter',
    )
    print(model_a.summary(slim=True, title='weight = a * diameter + b'))

    model_b = create_regression_stat_model(
        data,
        'Whole_weight ~ Diameter + I(Diameter**2)',
    )
    print(model_b.summary(slim=True, title='weight = a * diameter + b * diameter^2 + c'))

    model_c = create_regression_stat_model(
        data,
        'Whole_weight ~ I(Diameter**3)',
    )
    print(model_c.summary(slim=True, title='weight = a * diameter^3'))

    model_d = create_regression_stat_model(
        data,
        'log (Whole_weight) ~ Diameter',  # Warning: there's a space between `log` and `(Whole_weight)`
    )
    print(model_d.summary(slim=True, title='log(weight) = a * diameter + b'))

    plot_points(
        x_series=whole_weight,
        y_series=diameter,
        x_axis_label='Whole_weight',
        y_axis_label='Diameter',
        point_size=3,
    )

    show_plot(title='Whole_weight vs Diameter', should_show_grid=True)


def run() -> None:
    abalone = retrieve_uci_data(repo_name='abalone')
    run_problem3(dataset=abalone)


if __name__ == '__main__':
    run()
