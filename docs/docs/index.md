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

cards[0]
#  {'name': "Ancestor's Chosen",
#   'mana_cost': '{5}{W}{W}',
#   'mana_value': 7,
#   'types': ['Creature'],
#   'printing': '10E',
#   'rarity': 'uncommon',
#   'text': "First strike (This creature deals combat damage before creatures without first strike.)\nWhen Ancestor's Chosen enters the battlefield, you gain 1 life for each card in your graveyard.",
#   'power': '4',
#   'toughness': '4'}

# load as a pandas.DataFrame
cards = load_cards(as_dataframe=True)

cards.head()
#                 name  mana_cost  mana_value       types printing    rarity                                               text power toughness
# 0  Ancestor's Chosen  {5}{W}{W}           7  [Creature]      10E  uncommon  First strike (This creature deals combat damag...     4         4
# 1     Angel of Mercy     {4}{W}           5  [Creature]      10E  uncommon  Flying\nWhen Angel of Mercy enters the battlef...     3         3
# 2   Aven Cloudchaser     {3}{W}           4  [Creature]      10E    common  Flying (This creature can't be blocked except ...     2         2
# 3     Ballista Squad     {3}{W}           4  [Creature]      10E  uncommon  {X}{W}, {T}: Ballista Squad deals X damage to ...     2         2
# 4            Bandage        {W}           1   [Instant]      10E    common  Prevent the next 1 damage that would be dealt ...  None      None

# load cards with the latest 17lands stats
cards = load_cards(with_stats=True)

cards[0]["ever_drawn_win_rate"]
# 0.49520352482795715
```

That's it!  `mtglearn` will automatically cache the downloaded datasets to skip downloading/preprocessing for next time. `datasets` uses [PyArrow](https://arrow.apache.org/docs/python/index.html) for loading/saving datasets so it is super fast. Pass the `refresh=True` arg to bypass the cache and get the latest data.


## API

### Datasets

::: mtglearn.datasets.load_cards
    rendering:
        heading_level: 4
 
::: mtglearn.datasets.load_rules
    rendering:
        heading_level: 4
 
### Schemas

These schemas determine the fields and field types of the core `mtglearn` objects.  They also determine the features of the corresponding `datasets.Dataset` and columns of the `pandas.DataFrame`.

For example:

- A `mtglearn.datasets.Card` has a `str` field called `name` for the name of the card.
- This means that the `datasets.Dataset` you get from `load_cards()` will have a `datasets.Value("string")` feature called `name`.  
- The `pandas.DataFrame` you get from `load_cards(as_dataframe=True)` will have a `name` column with `dtype=object`.


::: mtglearn.datasets.Card
    rendering:
        heading_level: 4
    selection:
        filters:
            - "!^_"  # exlude all members starting with _


::: mtglearn.datasets.CardStats
    rendering:
        heading_level: 4
    selection:
        filters:
            - "!^_"  # exlude all members starting with _


::: mtglearn.datasets.CardWithStats
    rendering:
        heading_level: 4
        show_bases: false
    selection:
        filters:
            - "!^_"  # exlude all members starting with _



::: mtglearn.datasets.Rule
    rendering:
        heading_level: 4
    selection:
        filters:
            - "!^_"  # exlude all members starting with _





