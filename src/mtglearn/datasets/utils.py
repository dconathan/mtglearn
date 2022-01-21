from typing import List, Optional, Set
from datasets import Features, Value, Sequence
import attrs


def type2features(cls) -> Features:
    """
    A helper function for turning an attrs class into the corresponding dataset.Features object
    """

    # if Optional grab actual type
    if type(cls) is type(Optional[str]):
        cls = cls.__args__[0]

    # if list, sequence of inner type
    if type(cls) is type(List[str]):
        return Sequence(type2features(cls.__args__[0]))

    # scalars
    if cls is str:
        return Value("string")
    if cls is int:
        return Value("int32")
    if cls is float:
        return Value("float32")

    # if is an attrs class, recurse and update
    if attrs.has(cls):
        features = {}
        for field in attrs.fields(cls):
            field_feature = type2features(field.type)
            features[field.name] = field_feature
        return Features(**features)

    raise NotImplementedError(str(types))

