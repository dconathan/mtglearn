from typing import List
import json
from collections import defaultdict
import random
import re

import requests
from datasets.utils.file_utils import cached_path
from datasets.features import Features
from datasets import Value, Dataset, Sequence

import attrs
from attrs import define

import cattrs


RAW_DATA_URL = "https://mtgjson.com/api/v5/AllPrintings.json"


def cls2features(cls) -> Features:
    features = {}
    for field in attrs.fields(cls):
        if field.type is str:
            features[field.name] = Value("string")
        elif field.type is int:
            features[field.name] = Value("int32")
        elif field.type is float:
            features[field.name] = Value("float32")
        elif field.type is List[str]:
            features[field.name] = Sequence(Value("string"))
        else:
            raise NotImplementedError(field.type)
    return Features(**features)


@define
class Card:
    name: str = attrs.field(default="")
    mana_cost: str = attrs.field(default="")
    mana_value: int = attrs.field(default=None)
    types: List[str] = attrs.field(default=attrs.Factory(list))
    expansion: str = attrs.field(default="")
    rarity: str = attrs.field(default="")
    text: str = attrs.field(default="")
    power: str = attrs.field(default="")  # needs to be str because of `*`
    toughness: str = attrs.field(default="")  # needs to be str because of `*`

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        d.update(kwargs)
        d["mana_cost"] = d.get("mana_cost", d.get("manaCost", ""))
        d["mana_value"] = d.get("mana_value", d.get("manaValue"))
        return cattrs.structure(d, cls)

    def __str__(self):
        """Canonical string representation of a card:

        field1_name: field1_value | field2_name: field2_value | ..."""
        fields = []
        for f in attrs.fields(type(self)):
            value = getattr(self, f.name)
            if isinstance(value, list):
                value = " ".join(value)
            if value:
                fields.append((f.name, value))
        return re.sub(r"\s+", " ", " | ".join(f"{k}: {v}" for k, v in fields))


@define
class CardWithStats:
    # colors: List[str] = attrs.field(default=attrs.Factory(list))
    seen_count: int
    avg_seen: float
    pick_count: int
    game_count: int
    win_rate: float
    sideboard_game_count: int
    sideboard_win_rate: float
    drawn_game_count: int
    drawn_win_rate: float
    ever_drawn_game_count: int
    drawn_game_count: int
    drawn_win_rate: float
    ever_drawn_game_count: int
    ever_drawn_win_rate: float
    never_drawn_game_count: int
    never_drawn_win_rate: float
    drawn_improvement_win_rate: float


EXPANSIONS = ["VOW"]


def load_cards_with_stats():

    cards = load_cards()

    for expansion in EXPANSIONS:
        expansion_cards = cards.filter(lambda x: x["expansion"] == expansion)
        print(expansion_cards)
        break

        endpoint = f"https://www.17lands.com/card_ratings/data?expansion={expansion}&format=PremierDraft"
        raw_data = requests.get(endpoint).json()


def load_cards(
    as_attrs=False, as_dataset=False, as_dataframe=False, shuffle=False, seed=None
):

    # as_dataframe is the default
    if not (as_attrs or as_dataset):
        as_dataframe = True

    path = cached_path(RAW_DATA_URL)

    cards = []

    with open(path) as f:
        raw_data = json.load(f)["data"]
        for expansion_name, expansion_data in raw_data.items():
            for raw_card in expansion_data["cards"]:
                cards.append(Card.from_dict(raw_card, expansion=expansion_name))

    if shuffle:
        random.Random(seed).shuffle(cards)

    if as_attrs:
        return cards

    dataset = defaultdict(list)

    for card in cards:
        for k, v in cattrs.unstructure(card).items():
            dataset[k].append(v)

    features = cls2features(Card)

    dataset = Dataset.from_dict(dataset, features=features)

    if as_dataframe:
        return dataset.to_pandas()

    # as_dataset=True
    return dataset


if __name__ == "__main__":
    load_cards_with_stats()
