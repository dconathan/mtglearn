from mtglearn.datasets.cards import _process_raw_cards
from mtglearn.datasets import load_cards
from mtglearn.card import Card
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


def test_load_cards_as_attrs():

    cards = load_cards(as_attrs=True)

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


def test_load_as_attrs_with_stats():

    cards = load_cards(as_attrs=True, with_stats=True)
