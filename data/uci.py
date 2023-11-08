from typing import Final

from ucimlrepo import fetch_ucirepo, dotdict

from model.uciml import UciMLRepo


def retrieve_uci_data(*, repo_name: str) -> UciMLRepo:
    """
    Retrieve the UCI dataset from the UCI repository.
    Args:
        repo_name: name of the dataset in the UCI repository
    Returns:
        the dataset as a pandas dataframe
    """

    dataset: Final[dotdict] = fetch_ucirepo(name=repo_name)
    repo = UciMLRepo.from_dotdict(raw_dict=dataset)
    return repo
