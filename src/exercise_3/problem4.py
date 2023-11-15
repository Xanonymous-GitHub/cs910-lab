import numpy as np

from data.uci import retrieve_uci_data
from model.uciml import UciMLRepo
from plot.stats import create_logistic_stat_model


def run_problem4(*, dataset: UciMLRepo) -> None:
    data = dataset.data.original

    # TODO: use .loc
    data['Sex'] = np.where(data['Sex'] == 'I', 1, 0)
    print(data['Sex'].value_counts())

    model_a = create_logistic_stat_model(
        data,
        'Sex ~ Length',
    )
    print(model_a.summary(title='logit(Sex) = a * Length + b'))

    model_b = create_logistic_stat_model(
        data,
        'Sex ~ Whole_weight',
    )
    print(model_b.summary(title='logit(Sex) = a * Whole_weight + b'))

    model_c = create_logistic_stat_model(
        data,
        'Sex ~ Rings',
    )
    print(model_c.summary(title='logit(Sex) = a * Rings + b'))

    model_d = create_logistic_stat_model(
        data,
        'Sex ~ Length + Whole_weight + Rings',
    )
    print(model_d.summary(title='logit(Sex) = a * Length + b * Whole_weight + c * Rings + d'))


def run() -> None:
    abalone = retrieve_uci_data(repo_name='abalone')
    run_problem4(dataset=abalone)


if __name__ == '__main__':
    run()