from data.uci import retrieve_uci_data
from model.uciml import UciMLRepo
from plot.stats import create_regression_stat_model


def run_problem2(*, dataset: UciMLRepo) -> None:
    data = dataset.data.original

    model = create_regression_stat_model(
        data,
        'Whole_weight ~ Shucked_weight + Viscera_weight + Shell_weight',
    )
    print(model.summary(slim=True))
    print(f'R: {model.rsquared ** 0.5}')


def run() -> None:
    abalone = retrieve_uci_data(repo_name='abalone')
    run_problem2(dataset=abalone)


if __name__ == '__main__':
    run()
