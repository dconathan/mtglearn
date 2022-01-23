#
MtG Learn is a project for applying machine learning to the trading card game Magic: the Gathering.


## Installation

The `mtglearn` Python package is available from [PyPI](https://pypi.org/):

```
pip install mtglearn
```


## Quickstart

To use `mtglearn` to load datasets:

```
from mtglearn.datasets import load_cards

# by default, loads cards as a pandas DataFrame
cards = load_cards()

# load cards with the latest 17lands stats
cards = load_cards(with_stats=True)
```

That's it!  `mtglearn` will automatically cache the downloaded datasets to skip downloading/preprocessing each time.  Pass the `refresh=True` arg to bypass the cache and get the latest data. See the docs for more details.  See datasets for more datasets that are avialable.

## Roadmap


