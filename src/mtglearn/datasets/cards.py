from typing import List, Mapping, Optional
import json
from collections import defaultdict
import random
import re
import os
from functools import lru_cache
from types import MappingProxyType
import logging

import attrs
from attrs import define
import cattrs
from cattrs.gen import make_dict_unstructure_fn, make_dict_structure_fn, override
import requests
from datasets.utils.file_utils import cached_path
from datasets import Features, Value, Dataset, Sequence, load_from_disk

from ..config import MTGLEARN_CACHE_HOME
from ..card import Card, CardStats, CardWithStats
from .utils import type2features, try_load_dataset, check_args


logger = logging.getLogger(__name__)


RAW_DATA_URL = "https://mtgjson.com/api/v5/AllPrintings.json"
CARDS_DATASET_CACHE = os.path.join(MTGLEARN_CACHE_HOME, "cards")
CARD_STATS_DATASET_CACHE = os.path.join(MTGLEARN_CACHE_HOME, "card_stats")

PRINTINGS_WITH_STATS = {"VOW"}
BASIC_LANDS = {"Plains", "Mountain", "Swamp", "Island", "Forest"}


@lru_cache(2 ** 8)
def _get_seventeenlands_stats(
    printing: str, stats_format: str = "PremierDraft"
) -> Mapping[str, CardStats]:
    logger.info(f"getting 17lands stats for {printing}...")
    if printing not in PRINTINGS_WITH_STATS:
        return None
    endpoint = f"https://www.17lands.com/card_ratings/data?expansion={printing}&format={stats_format}"
    response = requests.get(endpoint)
    raw_seventeenlands_stats = response.json()
    if not raw_seventeenlands_stats:
        raise ValueError(f"17lands returned no stats for {printing} {stats_format}")

    seventeenlands_stats = cattrs.structure(raw_seventeenlands_stats, List[CardStats])
    # mapping proxy type is immutable, for cache purposes
    return MappingProxyType({c.name: c for c in seventeenlands_stats})


def _join_card_with_stats(card):
    # seveenlands cards are keyed by their front side
    seventeenlands_key = card["name"].split(" // ")[0]
    stats = _get_seventeenlands_stats(card["printing"])[seventeenlands_key]
    card.update(cattrs.unstructure(stats))
    return card


def _process_raw_cards(refresh=False):

    # make a custom cattrs "structure" function that renames fields according to field aliases
    rename = {}

    for field in attrs.fields(Card):
        if "alias" in field.metadata:
            rename[field.name] = override(rename=field.metadata["alias"])

    fromdict = make_dict_structure_fn(Card, cattrs.Converter(), **rename)

    todict = make_dict_unstructure_fn(Card, cattrs.Converter())

    path = cached_path(
        RAW_DATA_URL,
        cache_dir=MTGLEARN_CACHE_HOME,
        ignore_url_params=True,
        use_etag=False,
        force_download=refresh,
    )

    raw_dataset = defaultdict(list)

    with open(path) as f:
        raw_data = json.load(f)["data"]

    for printing_name, printing_data in raw_data.items():
        for raw_card in printing_data["cards"]:
            raw_card["printing"] = printing_name
            card = fromdict(raw_card)
            for k, v in todict(card).items():
                raw_dataset[k].append(v)

    dataset = Dataset.from_dict(raw_dataset, features=type2features(Card))

    dataset.save_to_disk(CARDS_DATASET_CACHE)

    return dataset


def load_cards(
    as_dataset: bool = False,
    as_objs: bool = False,
    as_dataframe: bool = False,
    with_stats: bool = False,
    refresh_cards: bool = False,
    refresh_stats: bool = False,
):
    """
    Hello world!
    """

    args = check_args(as_objs=as_objs, as_dataset=as_dataset, as_dataframe=as_dataframe)

    # try to load the Dataset object from cache
    if refresh_cards:
        dataset = None
    else:
        dataset = try_load_dataset(CARDS_DATASET_CACHE)

    # if None, download and process
    if dataset is None:
        dataset = _process_raw_cards(refresh=refresh_cards)

    # if with_stats, grab from cache or load from 17lands
    if with_stats:

        if refresh_stats:
            card_stats = None
        else:
            card_stats = try_load_dataset(CARD_STATS_DATASET_CACHE)

        # if None, download and process/join with dataset
        if card_stats is None:
            # filter out cards that won't have stats
            dataset = dataset.filter(lambda c: c["printing"] in PRINTINGS_WITH_STATS)
            dataset = dataset.filter(lambda c: c["name"] not in BASIC_LANDS)
            card_stats = dataset.map(
                _join_card_with_stats, features=type2features(CardWithStats)
            )
            # save to cache
            card_stats.save_to_disk(CARD_STATS_DATASET_CACHE)

        dataset = card_stats

    # if as_dataset, we are done
    if args.as_dataset:
        return dataset

    # convert dataset to pandas dataframe
    if args.as_dataframe:
        return dataset.to_pandas()

    # convert to attrs objects
    if args.as_objs:
        if with_stats:
            fromdict = make_dict_structure_fn(CardWithStats, cattrs.Converter())
        else:
            fromdict = make_dict_structure_fn(Card, cattrs.Converter())
        return [fromdict(c) for c in dataset]
