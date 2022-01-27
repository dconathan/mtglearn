from kfp.v2.dsl import component, Output, Dataset


@component(
    base_image="python:3.9-slim",
    packages_to_install=[
        "/gcs/mtglearn/packages/mtglearn-22.1.0.dev1-py3-none-any.whl"
    ],
)
def prepare_data(seed: int, n_duplicates: int, dataset: Output[Dataset]):

    from mtglearn.datasets import load_cards, load_rules
    from mtglearn.datasets.cards import Card
    from datasets import concatenate_datasets
    import attrs
    import cattrs
    import random

    # train_mlm component/script needs to have train data with .txt suffix
    dataset.path = dataset.path + ".txt"

    rng = random.Random(seed)
    cards = load_cards(as_dataset=True)
    rules = load_rules(as_dataset=True)

    def preprocess_card(batch):

        preprocessed_batch = {"text": []}

        # convert from dict of lists to list of dicts
        batch = [dict(zip(batch, _)) for _ in zip(*batch.values())]

        for card in batch:

            # count the number of non-empty fields
            fields = [k for k, v in card.items() if v]
            n_fields = len(fields)
            preprocessed_batch["text"].append(str(cattrs.structure(card, Card)))
            for _ in range(n_duplicates - 1):
                # n_fields - 1 because we always want to keep `name`
                n_fields_to_drop = rng.randint(0, n_fields - 1)
                candidate_fields = list(set(card.keys()).difference({"name"}))
                fields_to_keep = ["name"] + rng.sample(
                    candidate_fields, n_fields_to_drop
                )
                # modified card with missing values
                mcard = cattrs.structure({k: card[k] for k in fields_to_keep}, Card)
                preprocessed_batch["text"].append(str(mcard))

        return preprocessed_batch

    cards = cards.map(
        preprocess_card, batched=True, batch_size=1, remove_columns=cards.column_names
    )
    # filter out rules that are fewer than 5 tokens (likely just a heading)
    rules = rules.filter(lambda r: len(r["text"].split()) > 5)

    text = concatenate_datasets([cards, rules]).shuffle(seed=rng.randint(100, 1000))

    # write to output
    with open(dataset.path, "w") as f:
        for line in text:
            f.write(line["text"] + "\n")
