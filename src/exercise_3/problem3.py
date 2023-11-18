import numpy as np
from matplotlib import pyplot as plt

from data.uci import retrieve_uci_data
from model.uciml import UciMLRepo
from plot import plot_points, show_plot
from plot.stats import create_regression_stat_model


def run_problem3(*, dataset: UciMLRepo) -> None:
    data = dataset.data.original
    whole_weight = data['Whole_weight'].to_numpy()
    diameter = data['Diameter'].to_numpy()

    plot_points(
        x_series=diameter,
        y_series=whole_weight,
        x_axis_label='Diameter',
        y_axis_label='Whole_weight',
        point_size=3,
    )

    model_a = create_regression_stat_model(
        data,
        'Whole_weight ~ Diameter',
    )
    print(model_a.summary(slim=True, title='weight = a * diameter + b'))
    print(f'R: {model_a.rsquared ** 0.5}')
    painter_a = np.poly1d(model_a.params[::-1])
    poly_line_a = np.linspace(diameter.min(), diameter.max(), len(diameter))
    plt.plot(poly_line_a, painter_a(poly_line_a), '--', markersize=5, label='weight = a * diameter + b')

    model_b = create_regression_stat_model(
        data,
        'Whole_weight ~ Diameter + I(Diameter**2)',
    )
    print(model_b.summary(slim=True, title='weight = a * diameter + b * diameter^2 + c'))
    print(f'R: {model_b.rsquared ** 0.5}')
    painter_b = np.poly1d(model_b.params[::-1])
    poly_line_b = np.linspace(diameter.min(), diameter.max(), len(diameter))
    plt.plot(poly_line_b, painter_b(poly_line_b), '--', markersize=5,
             label='weight = a * diameter + b * diameter^2 + c')

    model_c = create_regression_stat_model(
        data,
        'Whole_weight ~ I(Diameter**3) -1',  # use `-1` to remove the constant
    )
    print(model_c.summary(slim=True, title='weight = a * diameter^3'))
    print(f'R: {model_c.rsquared ** 0.5}')
    a = model_c.params[0]
    # since this model doesn't have a constant, we need to add it manually
    # Note that this is a cubic function.
    b = whole_weight.mean() - a * diameter.mean() ** 3
    painter_c = np.poly1d([a, 0, 0, b])
    poly_line_c = np.linspace(diameter.min(), diameter.max(), len(diameter))
    plt.plot(poly_line_c, painter_c(poly_line_c), '--', markersize=5, label='weight = a * diameter^3')

    model_d = create_regression_stat_model(
        data,
        'log (Whole_weight) ~ Diameter',  # Warning: there's a space between `log` and `(Whole_weight)`
    )
    print(model_d.summary(slim=True, title='log(weight) = a * diameter + b'))
    print(f'R: {model_d.rsquared ** 0.5}')
    b, a = model_d.params
    poly_line_d = np.linspace(diameter.min(), diameter.max(), len(diameter))
    predicted_weight = np.exp(a * poly_line_d + b)
    plt.plot(poly_line_d, predicted_weight, '--', markersize=5, label='log(weight) = a * diameter + b')

    show_plot(title='Whole_weight vs Diameter', should_show_grid=True)


def run() -> None:
    abalone = retrieve_uci_data(repo_name='abalone')
    run_problem3(dataset=abalone)


if __name__ == '__main__':
    run()
