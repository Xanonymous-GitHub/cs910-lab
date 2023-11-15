from data.uci import retrieve_uci_data
from model.uciml import UciMLRepo
from plot.stats import create_logistic_stat_model
from utils.numeration import binary_quantified_from


def run_problem4(*, dataset: UciMLRepo) -> None:
    # TODO: use .loc
    data = binary_quantified_from(dataset.data.original, column='Sex', positive_when_equal_to='I')
    print(data['Sex'].value_counts())

    model_a, accuracy_a = create_logistic_stat_model(
        data,
        'Sex ~ Length',
    )
    print(model_a.summary(title='logit(Sex) = a * Length + b'))
    print(f'accuracy: {accuracy_a}')

    model_b, accuracy_b = create_logistic_stat_model(
        data,
        'Sex ~ Whole_weight',
    )
    print(model_b.summary(title='logit(Sex) = a * Whole_weight + b'))
    print(f'accuracy: {accuracy_b}')

    model_c, accuracy_c = create_logistic_stat_model(
        data,
        'Sex ~ Rings',
    )
    print(model_c.summary(title='logit(Sex) = a * Rings + b'))
    print(f'accuracy: {accuracy_c}')

    model_d, accuracy_d = create_logistic_stat_model(
        data,
        'Sex ~ Length + Whole_weight + Rings',
    )
    print(model_d.summary(title='logit(Sex) = a * Length + b * Whole_weight + c * Rings + d'))
    print(f'accuracy: {accuracy_d}')


def run() -> None:
    abalone = retrieve_uci_data(repo_name='abalone')
    run_problem4(dataset=abalone)


if __name__ == '__main__':
    run()
