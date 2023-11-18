from multiprocessing import Pool, cpu_count
from pprint import pprint

from pandas import DataFrame

from data.uci import retrieve_uci_data
from model.uciml import UciMLRepo
from plot.stats import create_logistic_stat_model
from utils.iters import create_combinations_from
from utils.numeration import binary_quantified_from


def build_model(data: DataFrame, left_col: str, *right_cols: str):
    illegal_name_converted_right_cols = [f'Q("{col}")' for col in right_cols]
    equation = f'{left_col} ~ {" + ".join(illegal_name_converted_right_cols)}'
    return create_logistic_stat_model(data, equation)


def get_all_headers_from(data: DataFrame) -> list[str]:
    # return [header.replace('-', '_') for header in data.columns.to_list()]
    return data.columns.to_list()


def prepare_data(*, dataset: UciMLRepo) -> DataFrame:
    data = dataset.data.original
    data = binary_quantified_from(data, column='sex', positive_when_equal_to='Male')

    # replace all '<=50K.' to '<=50K', and '>50K.' to '>50K' in 'income' column.
    data['income'] = data['income'].str.replace('.', '')

    # remove all rows that contains '?'.
    data = data[~data.eq('?')]

    # Show all value count of all columns.
    show_all_value_count_of(data)

    return data


def model_building_batch(
        data,
        left_col: str,
        batch_patterns: tuple[tuple[str]]
) -> [tuple[float, str, ...]]:
    results = []
    for pattern in batch_patterns:
        model, accuracy = build_model(data, left_col, *pattern)
        results.append((accuracy, *pattern))
    print(f'Batch done, size: {len(batch_patterns)}, signature: {hash(batch_patterns)}')
    return results


def create_batches[T](items, batch_size) -> [[T]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def show_all_value_count_of(data: DataFrame) -> None:
    for column in data.columns:
        print(f'{column}:')
        print(data[column].value_counts())
        print()


def find_best_accuracy(*, data: DataFrame) -> None:
    all_headers = get_all_headers_from(data)
    all_headers.remove('sex')

    equation_patterns = tuple(frozenset(create_combinations_from(all_headers)))
    batches = create_batches(equation_patterns, 100)
    accuracy_with_patterns = []

    max_cpu_count = cpu_count()

    with Pool(max_cpu_count) as pool:
        batch_results = pool.starmap(
            model_building_batch,
            [(data, 'sex', batch) for batch in batches]
        )

    for batch_result in batch_results:
        accuracy_with_patterns.extend(batch_result)

    accuracy_with_patterns.sort(key=lambda item: item[0], reverse=True)
    print('Best accuracy: ')
    pprint(accuracy_with_patterns[0])


def run_problem5(*, dataset: UciMLRepo) -> None:
    print(dataset.variables)

    data = prepare_data(dataset=dataset)
    all_headers = get_all_headers_from(data)
    all_headers.remove('sex')

    print('start building model...')
    model, accuracy = build_model(
        data,
        'sex',
        *all_headers
    )
    print(model.summary())
    print(f'accuracy: {accuracy}')

    should_start_finding_best_equation = input('Do you want to start finding the best equation? (y/n): ') == 'y'
    if not should_start_finding_best_equation:
        return

    find_best_accuracy(data=data)


def run() -> None:
    adult = retrieve_uci_data(repo_name='adult')
    # with open('data/adult.csv') as f:
    #     adult_csv = DataFrame(DictReader(f))
    # adult = adult.replace_original(adult_csv)
    run_problem5(dataset=adult)


if __name__ == '__main__':
    run()
