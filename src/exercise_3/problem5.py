from multiprocessing import Pool, cpu_count
from pprint import pprint

from pandas import DataFrame

from data.uci import retrieve_uci_data
from model.uciml import UciMLRepo
from plot.stats import create_logistic_stat_model
from utils.colors import ConsoleColorWrapper, ConsoleColors
from utils.iters import create_combinations_from
from utils.numeration import binary_quantified_from


def build_model(data: DataFrame, left_col: str, *right_cols: str):
    illegal_name_converted_right_cols = [f'Q("{col}")' for col in right_cols]
    equation = f'{left_col} ~ {" + ".join(illegal_name_converted_right_cols)}'
    return create_logistic_stat_model(data, equation)


def get_all_headers_from(data: DataFrame) -> list[str]:
    # return [header.replace('-', '_') for header in data.columns.to_list()]
    return data.columns.to_list()


def prepare_data(*, data: DataFrame) -> DataFrame:
    data = binary_quantified_from(data, column='sex', positive_when_equal_to='Male')

    # data = quantified_from(
    #     data,
    #     # 'workclass',
    #     # 'education',
    #     # 'marital-status',
    #     # 'occupation',
    #     # 'relationship',
    #     # 'race',
    #     # 'native-country'
    # )

    # replace all '<=50K.' to '<=50K', and '>50K.' to '>50K' in 'income' column.
    data['income'] = data['income'].str.replace('.', '')

    # remove all rows that contains '?'.
    data = data[~data.eq('?')]

    # Show all value count of all columns.
    # show_all_value_count_of(data)

    return data


def model_building_batch(
        data,
        left_col: str,
        batch_patterns: tuple[tuple[str]]
) -> [tuple[float, str, ...]]:
    results = []
    for pattern in batch_patterns:
        try:
            model, accuracy = build_model(data, left_col, *pattern)
        except Exception:
            # with ConsoleColorWrapper(ConsoleColors.GRAY):
                # print(f'\nfailed to build model with pattern: {pattern}')
            continue
        results.append((accuracy, *pattern))

    # show a gray dot to indicate that this batch is done.
    with ConsoleColorWrapper(ConsoleColors.CYAN):
        print('.', end='')
    return results


def create_batches[T](items, batch_size) -> [[T]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def show_all_value_count_of(data: DataFrame) -> None:
    for column in data.columns:
        print(f'{column}:')
        print(data[column].value_counts())
        print('')


def batch_model_build_process(*, data: DataFrame, batches) -> [tuple[float, str, ...]]:
    accuracy_with_patterns = []

    max_cpu_count = cpu_count()

    with Pool(max_cpu_count) as pool:
        batch_results = pool.starmap(
            model_building_batch,
            [(data, 'sex', batch) for batch in batches]
        )

    for batch_result in batch_results:
        accuracy_with_patterns.extend(batch_result)

    return accuracy_with_patterns


def find_best_accuracy(*, data: DataFrame) -> None:
    all_headers = get_all_headers_from(data)
    all_headers.remove('sex')

    equation_patterns = tuple(frozenset(create_combinations_from(all_headers)))
    batches = create_batches(equation_patterns, 1000)
    accuracy_with_patterns = batch_model_build_process(data=data, batches=batches)

    accuracy_with_patterns.sort(key=lambda item: item[0], reverse=True)

    with ConsoleColorWrapper(ConsoleColors.GREEN):
        print('\nbest accuracy:')
    pprint(accuracy_with_patterns[0])


def find_accuracy_affection(*, data: DataFrame, all_headers: frozenset[str], max_accuracy: float) -> None:
    """
    find the affection of each variable on the accuracy.
    It will try to remove each variable from the equation and see how the accuracy changes.
    It will continue to find the affection until any pattern cause the accuracy below 1% of the max accuracy.
    """
    expected_combination_length = (len_of_all := len(all_headers)) - 1
    between_max_accuracy = 0
    results: [tuple[float, float, str]] = []

    zero_to_one = 0
    while zero_to_one < 1:
        zero_to_one += 1

        equation_patterns = tuple(frozenset(
            create_combinations_from(
                list(all_headers),
                min_length=expected_combination_length,
                max_length=expected_combination_length
            )
        ))

        with ConsoleColorWrapper(ConsoleColors.CYAN):
            print(f'\nremove {len_of_all - expected_combination_length} variables from the equation...')

        batches = create_batches(equation_patterns, 100)
        accuracy_with_patterns = batch_model_build_process(data=data, batches=batches)

        for accuracy, *pattern in accuracy_with_patterns:
            results.append((accuracy, max_accuracy - accuracy, all_headers - frozenset(pattern)))

        results.sort(key=lambda item: item[1], reverse=True)
        between_max_accuracy = results[0][1]
        expected_combination_length -= 1

    print()
    pprint(results)


def run_problem5(*, dataset: UciMLRepo) -> None:
    print(dataset.variables)

    data = prepare_data(data=dataset.data.original)
    all_headers = get_all_headers_from(data)
    all_headers.remove('sex')

    with ConsoleColorWrapper(ConsoleColors.CYAN):
        print('start building model...')
    model, accuracy = build_model(
        data,
        'sex',
        # *all_headers
        'occupation',
        'relationship',
    )
    print(model.summary())

    with ConsoleColorWrapper(ConsoleColors.GREEN):
        print(f'max accuracy: {accuracy}')

    # find_accuracy_affection(data=data, all_headers=frozenset(all_headers), max_accuracy=accuracy)

    with ConsoleColorWrapper(ConsoleColors.YELLOW):
        should_start_finding_best_equation = input('Do you want to start finding the best equation? (y/n): ') == 'y'
    if not should_start_finding_best_equation:
        return

    find_best_accuracy(data=data)


def run() -> None:
    adult = retrieve_uci_data(repo_name='adult')
    run_problem5(dataset=adult)


if __name__ == '__main__':
    run()
