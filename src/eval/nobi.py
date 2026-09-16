"""NOBI label conversion for nested single-token or multi-token spans.

NOBI keeps the outer BIO span and marks a nested span with ``BN``/``IN``.
The functions here operate on one sentence and return the same exclusive-end
span representation used by ``src.eval.spans``. They do not search the gold
term lists or decide which occurrences are annotated; callers must provide the
spans to encode.
"""

from __future__ import annotations

Span = tuple[int, int]

NOBI_LABELS = frozenset({"O", "B", "I", "BN", "IN"})


def _validate_spans(tokens: list[str], spans: list[Span]) -> None:
	for start, end in spans:
		if not 0 <= start < end <= len(tokens):
			raise AssertionError(f"invalid span {(start, end)} for {len(tokens)} tokens")


def encode_nobi(
	tokens: list[str],
	outer_spans: list[Span],
	nested_spans: list[Span],
) -> list[str]:
	"""Encode outer BIO spans plus nested spans as NOBI labels.

	``nested_spans`` must be strictly inside an outer span. Nested spans may
	contain one or more tokens; a one-token span is represented by ``BN``.
	Overlapping nested spans are rejected because one label cannot represent
	two nested spans beginning at the same token.
	"""
	_validate_spans(tokens, [*outer_spans, *nested_spans])
	labels = ["O"] * len(tokens)

	for start, end in outer_spans:
		for index in range(start, end):
			wanted = "B" if index == start else "I"
			if labels[index] != "O":
				raise AssertionError(f"overlapping outer spans at token {index}")
			labels[index] = wanted

	occupied_nested: set[int] = set()
	for nested_start, nested_end in nested_spans:
		containing = any(
			outer_start <= nested_start and nested_end <= outer_end
			for outer_start, outer_end in outer_spans
		)
		if not containing:
			raise AssertionError(f"nested span {(nested_start, nested_end)} is outside an outer span")

		for index in range(nested_start, nested_end):
			if index in occupied_nested:
				raise AssertionError(f"overlapping nested spans at token {index}")
			if labels[index] == "O":
				raise AssertionError(f"nested span {(nested_start, nested_end)} is outside the labels")
			labels[index] = "BN" if index == nested_start else "IN"
			occupied_nested.add(index)

	return labels


def decode_nobi(tokens: list[str], labels: list[str]) -> list[Span]:
	"""Decode NOBI labels into outer and nested spans.

	The result contains both kinds of spans. ``B``/``I``/``BN``/``IN`` form
	the outer span; ``BN``/``IN`` additionally form a nested span. A ``BN``
	not followed by ``IN`` is therefore a valid one-token nested term.
	"""
	if len(tokens) != len(labels):
		raise AssertionError(f"{len(tokens)} tokens vs {len(labels)} labels")
	unknown = set(labels) - NOBI_LABELS
	if unknown:
		raise ValueError(f"unknown NOBI labels: {sorted(unknown)}")

	spans: list[Span] = []
	outer_start: int | None = None
	nested_start: int | None = None

	def close_nested(end: int) -> None:
		nonlocal nested_start
		if nested_start is not None:
			spans.append((nested_start, end))
			nested_start = None

	def close_outer(end: int) -> None:
		nonlocal outer_start
		close_nested(end)
		if outer_start is not None:
			spans.append((outer_start, end))
			outer_start = None

	for index, label in enumerate(labels):
		if label == "O":
			close_outer(index)
		elif label == "B":
			close_outer(index)
			outer_start = index
		elif label == "I":
			if outer_start is None:
				raise ValueError(f"I at index {index} has no outer span")
			close_nested(index)
		elif label == "BN":
			if outer_start is None:
				raise ValueError(f"BN at index {index} has no outer span")
			close_nested(index)
			nested_start = index
		elif label == "IN":
			if outer_start is None:
				raise ValueError(f"IN at index {index} has no outer span")
			if nested_start is None:
				raise ValueError(f"IN at index {index} has no nested span")

	close_outer(len(labels))
	return spans
