from mtglearn.datasets import load_rules


def test_load_rules():

    load_rules()
    load_rules(as_dataset=True)
    load_rules(as_attrs=True)
    load_rules(as_dataframe=True)
