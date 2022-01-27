from typing import List, Optional, Set, Tuple
import os
import logging
from collections import defaultdict

from datasets import Dataset, Features, Value, Sequence, load_from_disk
import attrs
from attrs import frozen
import cattrs


logger = logging.getLogger(__name__)


def from_list(cls, xs: list) -> Dataset:
    """
    Helper function for turning a list of attrs attrs into the corresponding Dataset object
    """

    todict = cattrs.gen.make_dict_unstructure_fn(cls, cattrs.Converter())
    raw = defaultdict(list)
    for x in xs:
        for k, v in todict(x).items():
            raw[k].append(v)
    return Dataset.from_dict(raw, features=type2features(cls))


def type2features(cls) -> Features:
    """
    A helper function for turning an attrs class into the corresponding dataset.Features object
    """

    # if Optional grab actual type
    if type(cls) is type(Optional[str]):
        cls = cls.__args__[0]

    # if list, sequence of inner type
    if type(cls) is type(List[str]):
        return Sequence(type2features(cls.__args__[0]))

    # scalars
    if cls is str:
        return Value("string")
    if cls is int:
        return Value("int32")
    if cls is float:
        return Value("float32")

    # if is an attrs class, recurse and update
    if attrs.has(cls):
        features = {}
        for field in attrs.fields(cls):
            field_feature = type2features(field.type)
            features[field.name] = field_feature
        return Features(**features)

    raise NotImplementedError(str(types))


def try_load_dataset(filename: str) -> Optional[Dataset]:
    """
    Tries to load the dataset from a cache. Returns None if the path does not exist or if there was an error.
    """
    if os.path.exists(filename):
        try:
            dataset = load_from_disk(filename)
            logger.debug(f"loaded cached dataset from {filename}")
            return dataset
        except Exception as e:
            logger.error(f"could not load dataset from {filename}: {e}")
    return None


@frozen
class Args:
    as_attrs: bool
    as_dataset: bool
    as_dataframe: bool


def check_args(as_attrs: bool, as_dataset: bool, as_dataframe: bool) -> Args:

    if all([as_attrs, as_dataframe, as_dataset]):
        # since as_dataset is default, this means we got multiple values
        raise ValueError("Only one of as_dataframe or as_attrs must be True")
    elif not any([as_attrs, as_dataframe, as_dataset]):
        logger.warning("Setting as_dataset=True")
        as_dataset = True
    elif sum([as_attrs, as_dataframe, as_dataset]) > 1:
        # this means a non default was passed in:
        as_dataset = False

    return Args(as_attrs=as_attrs, as_dataset=as_dataset, as_dataframe=as_dataframe)
