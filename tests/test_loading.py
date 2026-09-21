"""Unit tests for the T2 corpus loader.

All tests run on hand-written synthetic fixtures in ``tests/fixtures/`` or on
throwaway trees built under ``tmp_path``. Nothing here touches the real ACTER
data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.statistics.loading import (
    DataConfig,
    annotation_dir,
    derive_text_path,
    inventory_ratios,
    list_annotation_files,
    load_domain,
    parse_annotation_file,
)
from src.statistics.s01_lengths import percentile, summarise_lengths

FIXTURES = Path(__file__).parent / "fixtures"


def _lengths(sentences):
    return [len(tokens) for tokens, _labels in sentences]


# --------------------------------------------------------------------------- #
# parse_annotation_file -- boundary and format rules
# --------------------------------------------------------------------------- #
def test_trailing_blank_line_at_eof_produces_no_empty_tail():
    sentences = parse_annotation_file(FIXTURES / "trailing_blank_eof.tsv")
    assert _lengths(sentences) == [2, 1]
    assert sentences[-1] == (["gamma"], ["B"])


def test_two_consecutive_blank_lines_are_one_boundary():
    sentences = parse_annotation_file(FIXTURES / "consecutive_blanks.tsv")
    assert _lengths(sentences) == [2, 1]


def test_crlf_line_endings_leave_no_carriage_return_on_tokens_or_labels():
    sentences = parse_annotation_file(FIXTURES / "crlf.tsv")
    assert _lengths(sentences) == [2, 1]
    for tokens, labels in sentences:
        for token in tokens:
            assert "\r" not in token
        for label in labels:
            assert "\r" not in label
    assert sentences[0] == (["alpha", "beta"], ["B", "I"])
    assert sentences[1] == (["gamma"], ["O"])


def test_missing_tab_raises_with_path_and_line_number():
    with pytest.raises(AssertionError) as exc:
        parse_annotation_file(FIXTURES / "missing_tab.tsv")
    assert "missing_tab.tsv:2" in str(exc.value)


def test_unknown_label_raises_with_path_and_line_number():
    with pytest.raises(AssertionError) as exc:
        parse_annotation_file(FIXTURES / "bad_label.tsv")
    message = str(exc.value)
    assert "bad_label.tsv:2" in message
    assert "'X'" in message


def test_known_three_sentence_fixture_has_exact_content():
    sentences = parse_annotation_file(FIXTURES / "known_three_sentences.tsv")
    assert _lengths(sentences) == [3, 1, 2]
    assert sentences == [
        (["a", "b", "c"], ["B", "I", "I"]),
        (["d"], ["O"]),
        (["e", "f"], ["B", "O"]),
    ]


# --------------------------------------------------------------------------- #
# Loader -- annotation drives, text is derived, missing text crashes
# --------------------------------------------------------------------------- #
def _tokens_of(body: str) -> list[str]:
    return [line.split("\t")[0] for line in body.splitlines() if line.strip()]


def _make_tree(root: Path, domain: str, files: dict, *,
               with_texts=True, unannotated: dict = None) -> DataConfig:
    """Build a minimal ``<root>/en/<domain>/annotated/...`` tree.

    ``files``       maps ``file_id`` -> annotation-file text. The matching
                    ``texts/`` and ``texts_tokenised/`` files are derived from
                    it (all tokens, space-joined onto one line).
    ``unannotated`` optional ``name`` -> content under ``unannotated_texts/``.
    """
    base = root / "en" / domain
    ann_dir = (base / "annotated" / "annotations" / "sequential_annotations"
               / "iob_annotations" / "without_named_entities")
    ann_dir.mkdir(parents=True)
    (base / "annotated" / "texts").mkdir(parents=True)
    (base / "annotated" / "texts_tokenised").mkdir(parents=True)

    for file_id, body in files.items():
        (ann_dir / f"{file_id}_seq_terms.tsv").write_text(body, encoding="utf-8")
        text = " ".join(_tokens_of(body)) + "\n"
        if with_texts:
            (base / "annotated" / "texts" / f"{file_id}.txt").write_text(text, encoding="utf-8")
        (base / "annotated" / "texts_tokenised" / f"{file_id}.txt").write_text(text, encoding="utf-8")

    if unannotated:
        udir = base / "unannotated_texts"
        udir.mkdir(parents=True)
        for name, content in unannotated.items():
            (udir / name).write_text(content, encoding="utf-8")

    return DataConfig(
        data_root=root,
        language="en",
        train_domains=(domain,),
        test_domain=domain,
        sequential_scheme="iob_annotations",
        ne_variant="without_named_entities",
    )


def test_load_domain_reads_documents_in_sorted_order(tmp_path):
    cfg = _make_tree(tmp_path, "xx", {
        "xx_en_02": "gamma\tB\n",
        "xx_en_01": "alpha\tB\nbeta\tO\n",
    })
    docs = load_domain("xx", cfg)
    assert [d.file_id for d in docs] == ["xx_en_01", "xx_en_02"]
    assert docs[0].sentences == [(["alpha", "beta"], ["B", "O"])]
    assert docs[1].sentences == [(["gamma"], ["B"])]


def test_annotation_with_absent_paired_text_raises(tmp_path):
    cfg = _make_tree(tmp_path, "xx", {"xx_en_01": "alpha\tB\n"}, with_texts=False)
    with pytest.raises(AssertionError) as exc:
        load_domain("xx", cfg)
    message = str(exc.value)
    assert "no paired text file" in message
    assert "xx_en_01.txt" in message


def test_derive_text_path_points_into_texts_dir(tmp_path):
    cfg = _make_tree(tmp_path, "xx", {"xx_en_01": "alpha\tB\n"})
    ann = list_annotation_files(cfg, "xx")[0]
    text_path = derive_text_path(cfg, "xx", ann)
    assert text_path == tmp_path / "en" / "xx" / "annotated" / "texts" / "xx_en_01.txt"


def test_stray_non_annotation_file_in_dir_is_ignored(tmp_path):
    cfg = _make_tree(tmp_path, "xx", {"xx_en_01": "alpha\tB\n"})
    (annotation_dir(cfg, "xx") / "README.txt").write_text("notes\n", encoding="utf-8")
    assert [p.name for p in list_annotation_files(cfg, "xx")] == ["xx_en_01_seq_terms.tsv"]


def test_wrong_scheme_path_trips_the_no_annotation_files_check(tmp_path):
    _make_tree(tmp_path, "xx", {"xx_en_01": "alpha\tB\n"})
    bad = DataConfig(tmp_path, "en", ("xx",), "xx", "io_annotations", "without_named_entities")
    with pytest.raises(AssertionError) as exc:
        list_annotation_files(bad, "xx")
    assert "no annotation files under" in str(exc.value)
    assert "io_annotations" in str(exc.value)


def test_domain_with_no_b_label_is_rejected_as_io_format(tmp_path):
    """iob_annotations/ and io_annotations/ are sibling directories. An IO file
    parses fine (I and O are valid) but has no B anywhere -- the domain-level
    check is what catches pointing the loader at the wrong one."""
    cfg = _make_tree(tmp_path, "xx", {
        "xx_en_01": "alpha\tI\nbeta\tO\n",
        "xx_en_02": "gamma\tO\n",
    })
    with pytest.raises(AssertionError) as exc:
        load_domain("xx", cfg)
    assert "no B label" in str(exc.value)


def test_domain_b_check_is_corpus_wide_not_per_file(tmp_path):
    """A single term-free file is fine as long as some file in the domain has
    a B."""
    cfg = _make_tree(tmp_path, "xx", {
        "xx_en_01": "nothing\tO\nhere\tO\n",
        "xx_en_02": "a\tB\nterm\tI\n",
    })
    docs = load_domain("xx", cfg)
    assert [d.file_id for d in docs] == ["xx_en_01", "xx_en_02"]


# --------------------------------------------------------------------------- #
# Integrity check -- inventory ratio
# --------------------------------------------------------------------------- #
def test_inventory_ratio_passes_when_most_of_the_corpus_is_unannotated(tmp_path):
    # paired stream: 4 tokens; unannotated: 96 tokens -> wind ratio 0.04
    cfg = _make_tree(
        tmp_path, "wind",
        {"wind_en_01": "a\tB\nb\tO\nc\tO\nd\tO\n"},
        unannotated={"extra.txt": ("w " * 96).strip() + "\n"},
    )
    docs = load_domain("wind", cfg)
    ratios = inventory_ratios(cfg, {"wind": docs})
    paired, corpus, ratio = ratios["wind"]
    assert (paired, corpus) == (4, 100)
    assert ratio == 0.04


def test_inventory_ratio_raises_for_wind_when_unannotated_corpus_is_loaded(tmp_path):
    # No unannotated_texts/ dir -> corpus == paired -> ratio 1.0 >= 0.30
    cfg = _make_tree(tmp_path, "wind", {"wind_en_01": "a\tB\nb\tO\n"})
    docs = load_domain("wind", cfg)
    with pytest.raises(RuntimeError) as exc:
        inventory_ratios(cfg, {"wind": docs})
    message = str(exc.value)
    assert "wind" in message
    assert "1.0000" in message


def test_inventory_ratios_asserts_wind_is_present(tmp_path):
    """Without wind, the only blocking bound is silently never evaluated."""
    cfg = _make_tree(tmp_path, "htfl", {"htfl_en_001": "a\tB\nb\tO\n"})
    docs = load_domain("htfl", cfg)
    with pytest.raises(AssertionError) as exc:
        inventory_ratios(cfg, {"htfl": docs})
    assert "wind missing" in str(exc.value)


# --------------------------------------------------------------------------- #
# percentile -- exact nearest-rank definition
# --------------------------------------------------------------------------- #
def test_percentile_nearest_rank_on_one_to_ten():
    values = list(range(1, 11))          # already ascending, n = 10
    assert percentile(values, 50) == 5   # ceil(5.0)  -> rank 5  -> idx 4
    assert percentile(values, 90) == 9   # ceil(9.0)  -> rank 9  -> idx 8
    assert percentile(values, 95) == 10  # ceil(9.5)  -> rank 10 -> idx 9
    assert percentile(values, 99) == 10  # ceil(9.9)  -> rank 10 -> idx 9


def test_percentile_clamps_and_handles_singletons():
    assert percentile([42], 0) == 42
    assert percentile([42], 100) == 42
    assert percentile([1, 2, 3], 100) == 3
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarise_lengths_basic():
    summary = summarise_lengths([1, 1, 2, 3, 10])
    assert summary["n_sentences"] == 5
    assert summary["n_tokens"] == 17
    assert summary["min"] == 1
    assert summary["max"] == 10
    assert summary["n_sentences_le_2_tokens"] == 3
    assert summary["p50"] == 2


def test_summarise_lengths_raises_on_empty():
    with pytest.raises((ValueError, IndexError)):
        summarise_lengths([])
