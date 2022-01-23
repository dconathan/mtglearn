from mtglearn.datasets.cards import _process_raw_cards, Card
from mtglearn.datasets.rules import Rule
from mtglearn.datasets import load_cards, load_rules
import pandas as pd
from datasets import Dataset


def test_process_cards():

    _process_raw_cards()


def test_load_cards_default():

    cards = load_cards()

    assert isinstance(cards, pd.DataFrame)
    assert len(cards) > 1
    assert len(cards.columns)


def test_load_cards_as_dataframe():

    cards = load_cards(as_dataframe=True)

    assert isinstance(cards, pd.DataFrame)
    assert len(cards) > 1
    assert len(cards.columns)


def test_load_cards_as_objs():

    cards = load_cards(as_objs=True)

    assert isinstance(cards, list)
    assert len(cards) > 1
    assert isinstance(cards[0], Card)


def test_load_cards_as_dataset():

    cards = load_cards(as_dataset=True)

    assert isinstance(cards, Dataset)
    assert len(cards) > 1
    assert isinstance(cards[0], dict)
    assert cards[0]


def test_load_with_stats():

    cards = load_cards(with_stats=True)


def test_load_as_dataframe_with_stats():

    cards = load_cards(as_dataframe=True, with_stats=True)


def test_load_as_dataset_with_stats():

    cards = load_cards(as_dataset=True, with_stats=True)


def test_load_as_objs_with_stats():

    cards = load_cards(as_objs=True, with_stats=True)


def test_load_rules():

    load_rules()
    load_rules(as_dataset=True)
    load_rules(as_objs=True)
    load_rules(as_dataframe=True)
