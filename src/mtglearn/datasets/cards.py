from typing import List, Mapping, Optional
import json
from collections import defaultdict
import random
import re
import os
from functools import lru_cache
from types import MappingProxyType
import logging
import gzip

import attrs
from attrs import define, frozen
import cattrs
from cattrs.gen import make_dict_unstructure_fn, make_dict_structure_fn, override
from datasets.utils.file_utils import cached_path
from datasets import Features, Value, Dataset, Sequence, load_from_disk
from ftfy import fix_text

from ..config import MTGLEARN_CACHE_HOME
from .utils import type2features, try_load_dataset, check_args


logger = logging.getLogger(__name__)


RAW_DATA_URL = "https://mtgjson.com/api/v5/AllPrintings.json.zip"
CARDS_DATASET_CACHE = os.path.join(MTGLEARN_CACHE_HOME, "cards")
CARD_STATS_DATASET_CACHE = os.path.join(MTGLEARN_CACHE_HOME, "card_stats")

PRINTINGS_WITH_STATS = {
    "VOW",
    "MID",
    "AFR",
    "STX",
    "KHM",
    "ZLR",
    "KLR",
    "M21",
    "AKR",
    "IKO",
}
BASIC_LANDS = {"Plains", "Mountain", "Swamp", "Island", "Forest"}


@frozen(slots=False)
class Card:
    """
    An `mtglearn.Card` object that respresents a single card.  The source of data is [MTGJSON](https://www.MTGJSON.com), which [aggregates from several sources](https://www.mtgjson.com/faq/#where-does-the-data-come-from). According to MTGJSON, an *Atomic Card* is:

    > an oracle-like entity of a Magic: The Gathering card that only stores evergreen data about a card that would never change from printing to printing.


    Attributes:
        name: [The name of the card](https://www.mtgjson.com/data-models/card-atomic/#name)
        mana_cost:
        mana_value:
        types:
        printing:
        rarity:
        text:
        power:
        toughness:
    """

    # fields from mtgjson
    name: Optional[str] = attrs.field(default=None)
    mana_cost: Optional[str] = attrs.field(default=None, metadata={"alias": "manaCost"})
    mana_value: Optional[int] = attrs.field(
        default=None, metadata={"alias": "manaValue"}
    )
    types: Optional[List[str]] = attrs.field(default=None)
    printing: Optional[str] = attrs.field(default=None)
    rarity: Optional[str] = attrs.field(default=None)
    text: Optional[str] = attrs.field(default=None)
    power: Optional[str] = attrs.field(default=None)  # needs to be str because of `*`
    toughness: Optional[str] = attrs.field(
        default=None
    )  # needs to be str because of `*`

    def __str__(self):
        """Canonical string representation of a card:

        field1_name: field1_value | field2_name: field2_value | ..."""
        fields = []
        for f in attrs.fields(Card):
            if f.repr:
                value = getattr(self, f.name)
                if isinstance(value, list):
                    value = " ".join(value)
                if value:
                    fields.append((f.name, value))
        return fix_text(
            re.sub(r"\s+|_+", " ", " | ".join(f"{k}: {v}" for k, v in fields))
        )


@frozen(slots=False)
class CardStats:
    # fields from 17lands
    name: Optional[str] = attrs.field(default=None)
    stats_format: Optional[str] = attrs.field(default=None)
    stats_colors: Optional[str] = attrs.field(default=None)
    seen_count: Optional[int] = attrs.field(default=None)
    avg_seen: Optional[float] = attrs.field(default=None)
    avg_pick: Optional[float] = attrs.field(default=None)
    pick_count: Optional[int] = attrs.field(default=None)
    game_count: Optional[int] = attrs.field(default=None)
    win_rate: Optional[float] = attrs.field(default=None)
    sideboard_game_count: Optional[int] = attrs.field(default=None)
    sideboard_win_rate: Optional[float] = attrs.field(default=None)
    drawn_game_count: Optional[int] = attrs.field(default=None)
    drawn_win_rate: Optional[float] = attrs.field(default=None)
    ever_drawn_game_count: Optional[int] = attrs.field(default=None)
    drawn_game_count: Optional[int] = attrs.field(default=None)
    drawn_win_rate: Optional[float] = attrs.field(default=None)
    ever_drawn_game_count: Optional[int] = attrs.field(default=None)
    ever_drawn_win_rate: Optional[float] = attrs.field(default=None)
    never_drawn_game_count: Optional[int] = attrs.field(default=None)
    never_drawn_win_rate: Optional[float] = attrs.field(default=None)
    drawn_improvement_win_rate: Optional[float] = attrs.field(default=None)


@frozen(slots=False)
class CardWithStats(Card, CardStats):
    pass


def load_cards(
    as_dataset: bool = True,
    as_attrs: bool = False,
    as_dataframe: bool = False,
    with_stats: bool = False,
    refresh: bool = False,
    refresh_cards: bool = False,
    refresh_stats: bool = False,
):
    """

    Arguments:
        as_dataset: return cards as a `datasets.Dataset` object
        as_attrs: return cards as attrs objects
        as_dataframe: return cards as a `pandas.DataFrame`

    Example:

        ```python
        from mtglearn.datasets import load_cards

        # load as a datasets.Dataset
        cards = load_cards()

        # as a pandas.DataFrame
        cards = load_cards(as_dataframe=True)

        # with 17lands stats
        cards_with_stats = load_cards(with_stats=True)

        ```
    """

    if refresh:
        refresh_cards = True
        refresh_stats = True

    args = check_args(
        as_attrs=as_attrs, as_dataset=as_dataset, as_dataframe=as_dataframe
    )

    logger.debug(f"loading cards with args: {args}")

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
            # filter out alchemy cards
            dataset = dataset.filter(lambda c: not c["name"].startswith("A-"))
            card_stats = dataset.map(
                _join_card_with_stats,
                features=type2features(CardWithStats),
                batched=True,
                batch_size=1,
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
    if args.as_attrs:
        if with_stats:
            fromdict = make_dict_structure_fn(CardWithStats, cattrs.Converter())
        else:
            fromdict = make_dict_structure_fn(Card, cattrs.Converter())
        return [fromdict(c) for c in dataset]


@lru_cache(2 ** 8)
def _get_seventeenlands_stats(
    printing: str, stats_format: str = "PremierDraft"
) -> Mapping[str, CardStats]:
    logger.info(f"getting 17lands stats for {printing}...")
    if printing not in PRINTINGS_WITH_STATS:
        return None
    endpoint = f"https://www.17lands.com/card_ratings/data?expansion={printing}&format={stats_format}&start_date=2020-01-01"

    path = cached_path(
        endpoint,
        cache_dir=MTGLEARN_CACHE_HOME,
        ignore_url_params=False,
        use_etag=False,
        force_download=False,
    )
    with open(path) as f:
        raw_seventeenlands_stats = json.load(f)

    if not raw_seventeenlands_stats:
        logger.error(f"17lands returned no stats for {printing} {stats_format}")
        return MappingProxyType({})
    seventeenlands_stats = cattrs.structure(raw_seventeenlands_stats, List[CardStats])
    # mapping proxy type is immutable, for cache purposes
    return MappingProxyType({c.name: c for c in seventeenlands_stats})


def _join_card_with_stats(batch):
    # this assumes batch_size=1
    card = {k: v[0] for k, v in batch.items()}
    format_stats = _get_seventeenlands_stats(card["printing"])
    key = card["name"]
    if key not in format_stats:
        # 17lands keys flip cards by their front side
        flip_key = key.split(" // ")[0]
        if flip_key in format_stats:
            key = flip_key
        else:
            # 17 lands keys split cards like 'A /// B'
            key = key.replace(" // ", " /// ")
    if key in format_stats:
        stats = format_stats[key]
    else:
        logger.warning(f"could not get stats for card: {card['name']}, skipping...")
        return {f.name: [] for f in attrs.fields(CardWithStats)}
    card.update(cattrs.unstructure(stats))
    return {k: [v] for k, v in card.items()}


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
        force_download=True,
        force_extract=True,
        extract_compressed_file=True,
    )

    # returns the directory where the file was extracted, so join with filename
    path = os.path.join(path, "AllPrintings.json")

    raw_dataset = defaultdict(list)

    with open(path) as f:
        raw_data = json.load(f)["data"]

    seen = set()

    for printing_name, printing_data in raw_data.items():
        for raw_card in printing_data["cards"]:
            uid = (raw_card["name"], printing_name)
            if uid in seen:
                continue
            seen.add(uid)
            raw_card["printing"] = printing_name
            card = fromdict(raw_card)
            for k, v in todict(card).items():
                raw_dataset[k].append(v)

    dataset = Dataset.from_dict(raw_dataset, features=type2features(Card))

    dataset.save_to_disk(CARDS_DATASET_CACHE)

    return dataset
