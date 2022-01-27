from mtglearn.datasets import load_cards
from sklearn.preprocessing import MinMaxScaler

cards = load_cards(as_dataset=True, with_stats=True)
scaler = MinMaxScaler((-0.95, 0.95))

columns = ["avg_seen", "avg_pick", "win_rate"]


def collect_labels(card):
    return {"label": [card[c] for c in columns]}


cards = cards.map(collect_labels)
scaler.fit(cards["label"])


def scale_labels(batch):
    return {"label": scaler.transform(batch["label"])}


cards = cards.map(scale_labels, batched=True)

print(cards["label"][:10])
