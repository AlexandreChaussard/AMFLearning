import numbers
import typing

FeatureName = typing.Hashable
RegTarget = numbers.Number
ClfTarget = typing.Union[bool, str, int]
Target = typing.Union[ClfTarget, RegTarget]
Dataset = typing.Iterable[typing.Tuple[dict, typing.Any]]
Stream = typing.Iterator[typing.Tuple[dict, typing.Any]]
