from multiprocessing import Pool, cpu_count
from pprint import pprint

from pandas import DataFrame

from data.uci import retrieve_uci_data
from model.uciml import UciMLRepo
from plot.stats import create_logistic_stat_model
from utils.iters import create_combinations_from
from utils.numeration import quantified_from, binary_quantified_from


def build_model(data: DataFrame, left_col: str, *right_cols: str):
    illegal_name_converted_right_cols = [f'Q("{col}")' for col in right_cols]
    equation = f'{left_col} ~ {" + ".join(illegal_name_converted_right_cols)}'
    return create_logistic_stat_model(data, equation)


def prepare_data(*, dataset: UciMLRepo) -> DataFrame:
    data = quantified_from(
        dataset.data.original,
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'native-country'
    )
    data = binary_quantified_from(data, column='income', positive_when_equal_to='>50K')
    data = binary_quantified_from(data, column='sex', positive_when_equal_to='Male')

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


def run_problem5(*, dataset: UciMLRepo) -> None:
    data = prepare_data(dataset=dataset)
    all_headers = dataset.data.headers.to_list()
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


def run() -> None:
    abalone = retrieve_uci_data(repo_name='adult')
    run_problem5(dataset=abalone)


if __name__ == '__main__':
    run()
