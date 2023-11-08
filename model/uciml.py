from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Any

from pandas import Index, DataFrame
from ucimlrepo import dotdict


@dataclass(frozen=True)
class _UciMLData:
    features: DataFrame
    original: DataFrame
    targets: DataFrame
    headers: Index
    ids: Optional[Sequence[Any]]  # TODO: check if this is the right type


@dataclass(frozen=True)
class _UciMLMetadata:
    uci_id: int
    abstract: str
    additional_info: dotdict
    area: str
    characteristics: tuple[str, ...]
    creators: tuple[str, ...]
    data_url: str
    feature_types: tuple[str, ...]
    num_features: int
    num_instances: int
    repository_url: str
    target_col: tuple[str, ...]
    tasks: tuple[str, ...]
    year_of_dataset_creation: int


@dataclass(frozen=True)
class UciMLRepo:
    data: _UciMLData
    metadata: _UciMLMetadata
    variables: DataFrame

    @staticmethod
    def from_dotdict(*, raw_dict: dotdict) -> UciMLRepo:
        data = raw_dict.data
        metadata = raw_dict.metadata

        return UciMLRepo(
            data=_UciMLData(
                features=data.features,
                original=data.original,
                targets=data.targets,
                headers=data.headers,
                ids=data.ids,
            ),
            metadata=_UciMLMetadata(
                uci_id=metadata.uci_id,
                abstract=metadata.abstract,
                additional_info=metadata.additional_info,
                area=metadata.area,
                characteristics=tuple(metadata.characteristics),
                creators=tuple(metadata.creators),
                data_url=metadata.data_url,
                feature_types=tuple(metadata.feature_types),
                num_features=metadata.num_features,
                num_instances=metadata.num_instances,
                repository_url=metadata.repository_url,
                target_col=tuple(metadata.target_col),
                tasks=tuple(metadata.tasks),
                year_of_dataset_creation=metadata.year_of_dataset_creation,
            ),
            variables=DataFrame(raw_dict.variables),
        )
