"""Shared corpus loader for the T2 data-statistics items.

Scope is locked in ``docs/data_layout.md`` and ``Tasks.md`` and is not
re-derived here:

  * English only.
  * Domains: ``corp``, ``equi``, ``wind`` (train) and ``htfl`` (test).
  * Annotated portion only.
  * Sequential annotations, IOB scheme, ``without_named_entities`` labels.

Design rule: iteration starts from the ANNOTATION files. The paired text path
is *derived* from each annotation file and asserted to exist. An unannotated
text has no annotation file and therefore cannot enter the pipeline -- the
structure enforces it, and the explicit assertion turns a silent wrong-data
bug into a loud crash.

The loaded token stream comes from the annotation file only -- never from
``texts_tokenised/``. ``inventory_ratios`` reads the text directories, but
only to validate what was already loaded; the load itself never depends on it.

ACTER v1.5 is pinned at a tag and was inspected by hand, so the corpus cannot
change under us. The checks here do not guard against a malformed corpus;
they guard against *our* config or paths pointing at the wrong place -- the
failure named as the T2 risk in ``Tasks.md``, which yields plausible numbers
rather than a crash. Every one of them raises immediately; there is no
warn-and-continue path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Repo root = two levels up from this file (src/stats/loading.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "data.json"

VALID_LABELS = frozenset({"B", "I", "O"})
ANNOTATION_SUFFIX = "_seq_terms.tsv"

Sentence = tuple[list[str], list[str]]  # (tokens, labels) -- two parallel lists


def _check(condition: bool, message: str) -> None:
    """Loud assertion that -O cannot strip. Used for every invariant."""
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    language: str
    train_domains: tuple[str, ...]
    test_domain: str
    sequential_scheme: str          # e.g. "iob_annotations"
    ne_variant: str                 # e.g. "without_named_entities"

    @property
    def domains(self) -> tuple[str, ...]:
        """Train domains then the test domain, in a fixed order."""
        return (*self.train_domains, self.test_domain)


def load_config(config_path=None) -> DataConfig:
    """Load the data config from JSON. Data root and domain list live here,
    never hardcoded in a script."""
    config_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    _check(config_path.is_file(), f"config file not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    
    """
    what is difference between abs path and relative path : 
    abs path aka full path and it start with / 
    relative path simply not absloute and you can complete by other path to make it abs
    our code handle both cases if someone latter decide to 
    """
    data_root = Path(raw["data_root"])
    if not data_root.is_absolute():
        data_root = (_REPO_ROOT / data_root).resolve()

    return DataConfig(
        data_root=data_root,#usually will be ate-acter/data/raw/ACTER
        language=raw["language"], #english
        train_domains=tuple(raw["train_domains"]), #["corp", "equi", "wind"]
        test_domain=raw["test_domain"], #"htfl",
        sequential_scheme=raw["sequential_scheme"], #"iob_annotations"
        ne_variant=raw["ne_variant"], #"without_named_entities"
    )


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    file_id: str #this id of the file for example "corp_en_01" or "wind_en_04" <domain>_<lang>_<nn> followed by _seq_terms.tsv
    sentences: list[Sentence] #one (tokens, labels) pair per sentence -- two parallel lists
    annotation_path: Path #the TSV file one token per line, TAB, label, blank lines as boundaries. most important line for loader
    # no text_path field: nothing reads it. load_document still calls
    # derive_text_path for its existence assertion, it just doesn't store it.

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_tokens(self) -> int:
        return sum(len(tokens) for tokens, _labels in self.sentences)


# --------------------------------------------------------------------------- #
# Path resolution (annotation -> text, never the reverse)
# --------------------------------------------------------------------------- #
def _annotated_root(cfg: DataConfig, domain: str) -> Path:
    return cfg.data_root / cfg.language / domain / "annotated"


def _domain_root(cfg: DataConfig, domain: str) -> Path:
    return cfg.data_root / cfg.language / domain


def annotation_dir(cfg: DataConfig, domain: str) -> Path:
    # No is_dir() check: a wrong path makes the glob in list_annotation_files
    # return nothing, and its "no annotation files under {adir}" check fires
    # with this same path. One check, not two.
    return (_annotated_root(cfg, domain)
            / "annotations" / "sequential_annotations"
            / cfg.sequential_scheme / cfg.ne_variant)


def file_id_from_annotation(annotation_path: Path) -> str:
    name = annotation_path.name
    _check(
        name.endswith(ANNOTATION_SUFFIX),
        f"not an annotation file (want *{ANNOTATION_SUFFIX}): {annotation_path}",
    )
    return name[: -len(ANNOTATION_SUFFIX)]


def derive_text_path(cfg: DataConfig, domain: str, annotation_path: Path) -> Path:
    """Derive the paired raw-text path from an annotation file and assert it
    exists. On a pinned corpus a missing pair means the config points
    somewhere wrong."""
    file_id = file_id_from_annotation(annotation_path)
    text_path = _annotated_root(cfg, domain) / "texts" / f"{file_id}.txt"
    _check(
        text_path.is_file(),
        f"no paired text file for annotation {annotation_path}: expected {text_path}",
    )
    return text_path


def whitespace_tokens(text_path: Path) -> list[str]:
    """Whitespace-split the whole file, UTF-8, BOM stripped. Matches the
    dataset's own tokenisation rule (`texts_tokenised/` is whitespace-split)."""
    text = text_path.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.split()


def list_annotation_files(cfg: DataConfig, domain: str) -> list[Path]:
    """Glob the annotation directory for its sequential-annotation files, in
    sorted order (deterministic). The glob pattern means a stray file cannot
    enter, so no per-file name check is needed."""
    adir = annotation_dir(cfg, domain)  # stops at without_named_entities/
    entries = sorted(adir.glob(f"*{ANNOTATION_SUFFIX}"))
    _check(len(entries) > 0, f"no annotation files under {adir}")
    return entries


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_annotation_file(annotation_path: Path) -> list[Sentence]:
    """Parse one sequential annotation file into a list of ``(tokens, labels)``
    sentences -- two parallel lists per sentence.

    One token per line, TAB, label. A blank line ends a sentence; a run of
    blank lines and a trailing blank produce no empty sentence (a sentence is
    appended only when it has content). Each non-blank line must split into
    exactly 2 TAB fields, and the label must be one of {B, I, O} -- the
    per-line half of the IO-vs-IOB guard. ``newline=""`` keeps ``\\r`` on the
    line so the strip below is real on CRLF files.
    """
    sentences: list[Sentence] = []
    tokens: list[str] = []
    labels: list[str] = []

    with open(annotation_path, "r", encoding="utf-8", newline="") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\r\n")

            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                raise AssertionError(
                    f"{annotation_path}:{lineno}: want 2 tab fields, "
                    f"got {len(parts)}: {line!r}")
            if parts[1] not in VALID_LABELS:
                raise AssertionError(
                    f"{annotation_path}:{lineno}: label {parts[1]!r} not in {{B, I, O}}")
            tokens.append(parts[0])
            labels.append(parts[1])

    if tokens:
        sentences.append((tokens, labels))
    return sentences


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_document(cfg: DataConfig, domain: str, annotation_path: Path) -> Document:
    derive_text_path(cfg, domain, annotation_path)   # assert the paired text exists
    return Document(
        file_id=file_id_from_annotation(annotation_path),
        sentences=parse_annotation_file(annotation_path),
        annotation_path=annotation_path,
    )


def load_domain(domain: str, cfg: DataConfig = None) -> list[Document]:
    """Load every annotated document for ``domain``.

    Returns ``list[Document]``; each has ``.file_id`` and ``.sentences`` (one
    ``(tokens, labels)`` pair per sentence). Files are processed in sorted
    order so output is deterministic.
    """
    cfg = cfg or load_config()
    docs = [load_document(cfg, domain, p) for p in list_annotation_files(cfg, domain)]

    # iob_annotations/ and io_annotations/ are sibling directories with similar
    # names. In an IO file every line still parses (I and O are valid in both
    # schemes) and every span-based statistic comes out silently wrong. A real
    # IOB domain has at least one B somewhere; assert at domain level, since a
    # single file may legitimately contain no terms.
    _check(
        any("B" in labels for doc in docs for _tokens, labels in doc.sentences),
        f"domain {domain!r}: no B label in any of {len(docs)} files -- this "
        f"looks like an IO-format directory, not {cfg.sequential_scheme}",
    )
    return docs


def domain_token_count(documents: list[Document]) -> int:
    return sum(doc.n_tokens for doc in documents)


# --------------------------------------------------------------------------- #
# Integrity check: inventory ratio (paired vs whole-domain corpus)
# --------------------------------------------------------------------------- #
# data_layout.md section 1: the wind corpus is ~52k annotated words out of
# ~314k total (~17%); the rest is unannotated. If the paired stream is anywhere
# near the whole corpus, the unannotated files were loaded. No other domain
# needs a bound -- htfl is annotated in full, so its ratio is 1.0 by
# construction and checking it proves nothing.
WIND_MAX_INVENTORY_RATIO = 0.30


def _domain_corpus_token_count(cfg: DataConfig, domain: str) -> int:
    """Whitespace tokens over the whole domain corpus: the annotated files
    (``annotated/texts_tokenised/``) plus the unannotated remainder
    (``unannotated_texts/``, absent for htfl). Same unit as the paired count,
    so tokenisation cancels in the ratio."""
    total = 0
    for directory in (_annotated_root(cfg, domain) / "texts_tokenised",
                      _domain_root(cfg, domain) / "unannotated_texts"):
        if directory.is_dir():
            for p in sorted(directory.iterdir()):
                if p.is_file():
                    total += len(whitespace_tokens(p))
    # No total > 0 check: impossible on the pinned corpus, and if it somehow
    # happened the division in inventory_ratios raises ZeroDivisionError.
    return total


def inventory_ratios(cfg: DataConfig, documents_by_domain: dict) -> dict:
    """Return ``{domain: (paired_tokens, corpus_tokens, ratio)}``.

    Raises if wind's ratio reaches ``WIND_MAX_INVENTORY_RATIO`` -- the
    signature of having loaded the unannotated corpus.
    """
    out = {}
    for domain, docs in documents_by_domain.items():
        paired = domain_token_count(docs)
        corpus = _domain_corpus_token_count(cfg, domain)
        out[domain] = (paired, corpus, paired / corpus)

    _check(
        "wind" in out,
        "inventory_ratios: wind missing; the only blocking bound was never evaluated",
    )
    paired, corpus, ratio = out["wind"]
    if ratio >= WIND_MAX_INVENTORY_RATIO:
        raise RuntimeError(
            f"inventory-ratio check FAILED for wind: ratio {ratio:.4f} "
            f">= {WIND_MAX_INVENTORY_RATIO} (paired {paired} / corpus {corpus}) "
            f"-- the unannotated wind corpus was almost certainly loaded"
        )
    return out

