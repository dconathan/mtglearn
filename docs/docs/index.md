# Docs

This page is a reference for the `mtglearn` Python package.


## Installation

The `mtglearn` Python package is available from [PyPI](https://pypi.org/):

```
pip install mtglearn
```


## Quickstart

To use `mtglearn` to load datasets:

```python
from mtglearn.datasets import load_cards

# by default, loads cards as a datasets.Dataset 
cards = load_cards()

# load as a pandas.DataFrame
cards = load_cards(as_dataframe=True)

# load cards with the latest 17lands stats
cards = load_cards(with_stats=True)
```

That's it!  `mtglearn` will automatically cache the downloaded datasets to skip downloading/preprocessing for next time. `datasets` uses [PyArrow](https://arrow.apache.org/docs/python/index.html) for loading/saving datasets so it is super fast. Pass the `refresh=True` arg to bypass the cache and get the latest data.


### API


#### Schemas

These schemas determine the fields and field types of the core `mtglearn` objects.  They also determine the features of the corresponding `datasets.Dataset` and columns of the `pandas.DataFrame`.

For example:

- A `mtglearn.Card` has a `str` field called `name` for the name of the card.
- This means that the `datasets.Dataset` you get from `load_cards()` will have a `datasets.Value("string")` feature called `name`.  
- Similarly, the `pandas.DataFrame` you get from `load_cards(as_dataframe=True)` will have a `name` column with `dtype=object`.


::: mtglearn.Card
    rendering:
        heading_level: 5
    selection:
        filters:
            - "!^_"  # exlude all members starting with _


#### Datasets

::: mtglearn.datasets.load_cards



