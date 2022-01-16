from kfp.v2.dsl import Output, Dataset


def preprocess_dataset(seed: int, n_duplicates: int, dataset: Output[Dataset]):
    from mtglearn.datasets import load_cards
    from mtglearn.datasets.cards import Card
    import attrs
    import cattrs
    import random

    cards = load_cards(as_dataset=True, shuffle=True, seed=seed)

    rng = random.Random(seed)

    def preprocess_card(batch):

        preprocessed_batch = {"card": []}

        # convert from dict of lists to list of dicts
        batch = [dict(zip(batch, _)) for _ in zip(*batch.values())]

        for card in batch:

            # count the number of non-empty fields
            fields = [k for k, v in card.items() if v]
            n_fields = len(fields)
            preprocessed_batch["card"].append(str(cattrs.structure(card, Card)))
            for _ in range(n_duplicates - 1):
                # n_fields - 1 because we always want to keep `name`
                n_fields_to_drop = rng.randint(0, n_fields - 1)
                candidate_fields = list(set(card.keys()).difference({"name"}))
                fields_to_keep = ["name"] + rng.sample(
                    candidate_fields, n_fields_to_drop
                )
                # modified card, with missing values
                mcard = cattrs.structure({k: card[k] for k in fields_to_keep}, Card)
                preprocessed_batch["card"].append(str(mcard))

        return preprocessed_batch

    cards = cards.map(
        preprocess_card, batched=True, batch_size=1, remove_columns=cards.column_names
    ).shuffle()

    for card in cards[:100]["card"]:
        print(card)

    if dataset:
        cards.save_to_disk(dataset.path)


if __name__ == "__main__":
    preprocess_dataset(9129568, 2, None)
