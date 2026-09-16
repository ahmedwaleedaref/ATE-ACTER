"""NOBI label conversion for nested single-token or multi-token spans.

NOBI keeps the outer BIO span and marks a nested span with ``BN``/``IN``.
The functions here operate on one sentence and return the same exclusive-end
span representation used by ``src.eval.spans``. They do not search the gold
term lists or decide which occurrences are annotated; callers must provide the
spans to encode.
"""

from __future__ import annotations

Span = tuple[int, int]

NOBI_LABEL2ID = {"O": 0, "B": 1, "I": 2, "BN": 3, "IN": 4}
NOBI_ID2LABEL = {index: label for label, index in NOBI_LABEL2ID.items()}
NOBI_LABELS = frozenset(NOBI_LABEL2ID)


def _validate_spans(tokens: list[str], spans: list[Span]) -> None:
	for start, end in spans:
		if not 0 <= start < end <= len(tokens):
			raise AssertionError(f"invalid span {(start, end)} for {len(tokens)} tokens")


def resolve_nested_spans(
	tokens: list[str],
	outer_spans: list[Span],
	nested_spans: list[Span],
) -> list[Span]:
	"""Apply NOBI's left-to-right, longest-match-first resolution rule."""
	_validate_spans(tokens, [*outer_spans, *nested_spans])
	selected: list[Span] = []
	occupied: set[int] = set()

	for start, end in sorted(
		nested_spans,
		key=lambda span: (span[0], -(span[1] - span[0]), span[1]),
	):
		containing = [
			(outer_start, outer_end)
			for outer_start, outer_end in outer_spans
			if outer_start <= start and end <= outer_end
		]
		if not containing:
			continue
		# The outer B has absolute priority at its left boundary.
		if any(start == outer_start for outer_start, _outer_end in containing):
			continue
		if any(index in occupied for index in range(start, end)):
			continue
		selected.append((start, end))
		occupied.update(range(start, end))

	return sorted(selected)


def encode_nobi(
	tokens: list[str],
	outer_spans: list[Span],
	nested_spans: list[Span],
) -> list[str]:
	"""Encode outer BIO spans plus nested spans as NOBI labels.

	Nested spans outside an outer span, sharing its start, or overlapping a
	previously selected nested span are ignored according to the NOBI rules.
	Nested spans may contain one or more tokens; a one-token span is represented
	by ``BN``.
	"""
	_validate_spans(tokens, outer_spans)
	labels = ["O"] * len(tokens)

	for start, end in outer_spans:
		for index in range(start, end):
			wanted = "B" if index == start else "I"
			if labels[index] != "O":
				raise AssertionError(f"overlapping outer spans at token {index}")
			labels[index] = wanted

	for nested_start, nested_end in resolve_nested_spans(tokens, outer_spans, nested_spans):
		for index in range(nested_start, nested_end):
			labels[index] = "BN" if index == nested_start else "IN"

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
			if outer_start is not None:
				close_nested(index)
		elif label == "BN":
			if outer_start is not None:
				close_nested(index)
				nested_start = index
		elif label == "IN":
			if outer_start is not None and nested_start is not None:
				continue

	close_outer(len(labels))
	return spans
