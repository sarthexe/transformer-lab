"""
Prepare a word-level tokenized corpus for training a small transformer, built
from five public-domain Project Gutenberg books held as plain text files in
``backend/data/raw/``.

Pipeline
--------
1. Download the five books into ``raw/`` with urllib, skipping any file that
   is already on disk (so a second run works offline).
2. Strip the Gutenberg header/footer boilerplate, chapter headings and long
   blank-line runs from each book.
3. Segment each book into PASSAGES by gluing consecutive paragraphs together
   until the running word count passes PASSAGE_MAX_WORDS; drop passages under
   MIN_PASSAGE_WORDS words.
4. Normalize each passage: lowercase, split punctuation into standalone
   tokens, collapse repeated whitespace, append the literal token "<eos>".
5. Count token frequencies over the whole corpus and keep the VOCAB_SIZE - 3
   most frequent tokens as the "real" vocabulary.
6. Hold out every VAL_EVERY-th passage (index % VAL_EVERY == VAL_EVERY - 1)
   as validation, so both splits see every book. No shuffling.
7. Encode both splits and concatenate each into one flat id stream.
8. Write vocab.json, train.bin, val.bin and stats.json next to this file.

A "passage" plays exactly the role a "story" played when this script read
TinyStories: it is the unit that gets "<eos>" appended, the unit the
vocabulary is counted over, and the unit the train/val split is taken on. The
stats.json keys still call that unit a story.

Output file formats
-------------------
vocab.json
    JSON object with two mirrored views of the vocabulary::

        {
          "token_to_id": {"<unk>": 0, "<pad>": 1, "<eos>": 2, "the": 3, ...},
          "id_to_token": ["<unk>", "<pad>", "<eos>", "the", ...]
        }

    Reserved ids: 0 = "<unk>", 1 = "<pad>", 2 = "<eos>". Real tokens occupy
    ids 3 .. vocab_size - 1, ordered by descending corpus frequency.
    len(id_to_token) == len(token_to_id) == vocab_size (4096).

train.bin
    Raw little-endian binary stream of token ids, numpy dtype uint16, no
    header. It is a flat concatenation of every training passage, in corpus
    order, each passage terminated by its "<eos>" token (so passages can be
    re-segmented by splitting on id 2). Read it back with::

        import numpy as np
        ids = np.fromfile("train.bin", dtype=np.uint16)

val.bin
    Identical format to train.bin, for the validation split (every
    VAL_EVERY-th passage).

stats.json
    JSON object::

        {
          "vocab_size": 4096,
          "train_tokens": <int, number of ids in train.bin>,
          "val_tokens": <int, number of ids in val.bin>,
          "unk_rate_train": <float in [0, 1]>,
          "unk_rate_val": <float in [0, 1]>,
          "stories_kept": <int, passages kept>,
          "stories_dropped": <int, passages dropped for being too short>,
          "books": {"alice": <int>, "grimm": <int>, ...}
        }

Usage
-----
    python backend/data/prepare.py
    python backend/data/prepare.py --skip-download
    python backend/data/prepare.py --max-passages 500
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
from tqdm import tqdm


class Book(NamedTuple):
    """One Project Gutenberg book: its ebook number and local file stem."""

    ebook_id: int
    name: str

    @property
    def filename(self) -> str:
        return f"{self.name}.txt"

    @property
    def url(self) -> str:
        return f"https://www.gutenberg.org/files/{self.ebook_id}/{self.ebook_id}-0.txt"

    @property
    def alternate_url(self) -> str:
        """Second path form Gutenberg serves the same text under; some ebooks
        only exist here."""
        return f"https://www.gutenberg.org/cache/epub/{self.ebook_id}/pg{self.ebook_id}.txt"


BOOKS: tuple[Book, ...] = (
    Book(11, "alice"),
    Book(16, "peterpan"),
    Book(236, "jungle"),
    Book(2591, "grimm"),
    Book(1597, "andersen"),
)

# Gutenberg rejects the stock "Python-urllib/3.x" agent.
USER_AGENT: str = "transformer-lab/1.0 (educational corpus builder; urllib)"
DOWNLOAD_TIMEOUT: int = 60
MIN_BOOKS: int = 2

# A book must survive boilerplate stripping with at least this much text.
MIN_BOOK_CHARS: int = 10_000

# Passage segmentation: glue paragraphs together until the running word count
# exceeds PASSAGE_MAX_WORDS, then start a new passage. The target band is
# 60-150 words, but paragraphs are atomic and the closing one overshoots, so on
# the real corpus passages measure ~150-250 words (a single long paragraph can
# be longer still).
PASSAGE_MAX_WORDS: int = 150
MIN_PASSAGE_WORDS: int = 20

# Every VAL_EVERY-th passage is held out, interleaved across all books, so the
# validation split is not one book's tail.
VAL_EVERY: int = 20

VOCAB_SIZE: int = 4096

UNK_TOKEN: str = "<unk>"
PAD_TOKEN: str = "<pad>"
EOS_TOKEN: str = "<eos>"
SPECIAL_TOKENS: tuple[str, ...] = (UNK_TOKEN, PAD_TOKEN, EOS_TOKEN)
UNK_ID: int = 0
EOS_ID: int = 2
SPECIAL_TOKEN_SET: frozenset[str] = frozenset(SPECIAL_TOKENS)

# Punctuation that must become a standalone token: . , ! ? " ' ; : -
PUNCTUATION_RE: re.Pattern[str] = re.compile(r"""([.,!?"';:-])""")
WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")
PARAGRAPH_RE: re.Pattern[str] = re.compile(r"\n\s*\n")

START_MARKER: str = "*** START OF"
END_MARKER: str = "*** END OF"
HEADING_MAX_CHARS: int = 60
MAX_BLANK_RUN: int = 3

OUTPUT_DIR: Path = Path(__file__).resolve().parent
RAW_DIR: Path = OUTPUT_DIR / "raw"

TOP_TOKENS_TO_SHOW: int = 20


# ---------------------------------------------------------------------------
# 1. download
# ---------------------------------------------------------------------------


def fetch_text(url: str) -> str:
    """GET `url` with a non-default User-Agent and decode it as utf-8,
    replacing undecodable bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def download_book(book: Book) -> str | None:
    """Fetch one book's text, trying its alternate URL form if the primary one
    fails (404 for ebooks Gutenberg only files under /cache/epub/, but also
    the truncated reads the site hands out under load).

    Returns None when every candidate URL failed. Failures are logged, not
    raised, so one dead book cannot sink the whole corpus.
    """
    for url in (book.url, book.alternate_url):
        try:
            return fetch_text(url)
        except urllib.error.HTTPError as exc:
            print(f"  HTTP {exc.code:<4} {url}")
        except (OSError, http.client.HTTPException) as exc:
            # OSError covers URLError/timeouts; HTTPException covers
            # IncompleteRead and friends, which are not OSErrors.
            print(f"  error    {url} ({type(exc).__name__}: {exc})")
    return None


def download_books(dest: Path) -> list[Path]:
    """Download every book in BOOKS into `dest`, skipping files already there.

    Falls back to the book's alternate URL form when the primary one 404s, and
    keeps going when a single book cannot be fetched. Returns the paths that
    are available afterwards, in BOOKS order.

    Raises RuntimeError if fewer than MIN_BOOKS books ended up on disk.
    """
    dest.mkdir(parents=True, exist_ok=True)
    available: list[Path] = []

    for book in tqdm(BOOKS, desc="downloading", unit="book", dynamic_ncols=True):
        target: Path = dest / book.filename
        if target.exists() and target.stat().st_size > 0:
            print(f"  have    {book.filename:<14} {target.stat().st_size:>9,} bytes (skipping)")
            available.append(target)
            continue

        text: str | None = download_book(book)
        if text is None:
            print(f"  FAILED  {book.filename:<14} every URL failed, skipping this book")
            continue

        target.write_text(text, encoding="utf-8", newline="\n")
        print(f"  got     {book.filename:<14} {len(text):>9,} chars")
        available.append(target)

    require_enough_books(available, dest)
    return available


def existing_books(dest: Path) -> list[Path]:
    """Return the BOOKS files already present in `dest`, without downloading.

    Raises RuntimeError if fewer than MIN_BOOKS books are there.
    """
    available: list[Path] = [
        path
        for book in BOOKS
        if (path := dest / book.filename).exists() and path.stat().st_size > 0
    ]
    for path in available:
        print(f"  have    {path.name:<14} {path.stat().st_size:>9,} bytes")
    require_enough_books(available, dest)
    return available


def require_enough_books(available: list[Path], dest: Path) -> None:
    """Fail loudly when the corpus would be built from too few books."""
    if len(available) >= MIN_BOOKS:
        return
    present: set[Path] = set(available)
    missing: list[str] = [book.filename for book in BOOKS if dest / book.filename not in present]
    raise RuntimeError(
        f"only {len(available)} of {len(BOOKS)} books are available in {dest} "
        f"(need at least {MIN_BOOKS}); missing: {', '.join(missing)}. "
        "Check the network connection, or drop the .txt files in by hand and "
        "re-run with --skip-download."
    )


# ---------------------------------------------------------------------------
# 2. strip gutenberg boilerplate
# ---------------------------------------------------------------------------


def is_heading(line: str) -> bool:
    """True for short all-caps lines like "CHAPTER IV" or "THE END"."""
    stripped: str = line.strip()
    if not stripped or len(stripped) >= HEADING_MAX_CHARS:
        return False
    if not any(char.isalpha() for char in stripped):
        return False
    return stripped == stripped.upper()


def strip_gutenberg(raw: str) -> str:
    """Remove the Gutenberg header/footer, chapter headings and blank-line runs.

    Everything up to and including the "*** START OF" line is dropped, as is
    the "*** END OF" line and everything after it. A missing marker is a
    warning, not an error: the text is kept whole. Short all-caps heading lines
    are then dropped, and runs of MAX_BLANK_RUN or more blank lines collapse to
    a single blank line.
    """
    lines: list[str] = raw.splitlines()

    start: int = 0
    for index, line in enumerate(lines):
        if line.lstrip().startswith(START_MARKER):
            start = index + 1
            break
    else:
        print(f"WARNING: no '{START_MARKER}' marker found; keeping the whole header")

    end: int = len(lines)
    for index in range(start, len(lines)):
        if lines[index].lstrip().startswith(END_MARKER):
            end = index
            break
    else:
        print(f"WARNING: no '{END_MARKER}' marker found; keeping the whole footer")

    body: list[str] = [line for line in lines[start:end] if not is_heading(line)]

    # Buffer each blank run and emit it only once its full length is known, so
    # a long run collapses to exactly one blank line and a short one survives
    # intact. A trailing run is dropped by the final strip.
    kept: list[str] = []
    blank_run: int = 0
    for line in body:
        if not line.strip():
            blank_run += 1
            continue
        if blank_run:
            kept.extend([""] * (1 if blank_run >= MAX_BLANK_RUN else blank_run))
            blank_run = 0
        kept.append(line)

    return "\n".join(kept).strip("\n")


def read_book(path: Path) -> str:
    """Read `path` as utf-8 and strip its Gutenberg boilerplate.

    Raises ValueError, naming the file, if too little text survives.
    """
    raw: str = path.read_text(encoding="utf-8", errors="replace")
    text: str = strip_gutenberg(raw)
    if len(text) < MIN_BOOK_CHARS:
        raise ValueError(
            f"{path.name}: only {len(text):,} chars survived boilerplate stripping, "
            f"expected at least {MIN_BOOK_CHARS:,} -- the download is probably "
            "truncated or the markers moved"
        )
    return text


# ---------------------------------------------------------------------------
# 3. segment into passages
# ---------------------------------------------------------------------------


def segment_passages(
    text: str,
    max_words: int = PASSAGE_MAX_WORDS,
    min_words: int = MIN_PASSAGE_WORDS,
) -> tuple[list[str], int]:
    """Split `text` into paragraphs on blank lines and glue consecutive
    paragraphs into passages.

    A passage closes as soon as its running word count exceeds `max_words`.
    Since the closing paragraph is not split, passages overshoot `max_words`
    by up to one paragraph; nothing closes below ~60 words. Passages under
    `min_words` words -- in practice only a book's trailing remnant -- are
    dropped.

    Returns (passages, dropped_count). Order is preserved: no shuffling.
    """
    paragraphs: list[str] = [
        cleaned
        for chunk in PARAGRAPH_RE.split(text)
        if (cleaned := WHITESPACE_RE.sub(" ", chunk).strip())
    ]

    passages: list[str] = []
    dropped: int = 0
    current: list[str] = []
    words: int = 0

    def flush() -> None:
        nonlocal dropped, words
        if not current:
            return
        if words < min_words:
            dropped += 1
        else:
            passages.append(" ".join(current))
        current.clear()
        words = 0

    for paragraph in paragraphs:
        current.append(paragraph)
        words += len(paragraph.split())
        if words > max_words:
            flush()
    flush()

    return passages, dropped


def collect_passages(paths: Iterable[Path]) -> tuple[list[tuple[str, str]], int]:
    """Read every book and segment it into passages.

    Returns (records, dropped_count), where each record is (book_name, passage)
    in book order and then in-book order.
    """
    records: list[tuple[str, str]] = []
    dropped: int = 0

    for path in tqdm(list(paths), desc="segmenting", unit="book", dynamic_ncols=True):
        passages, book_dropped = segment_passages(read_book(path))
        dropped += book_dropped
        records.extend((path.stem, passage) for passage in passages)
        print(f"  {path.stem:<14} {len(passages):>7,} passages ({book_dropped} dropped)")

    return records, dropped


# ---------------------------------------------------------------------------
# 4. normalize, count, encode
# ---------------------------------------------------------------------------


def normalize_passage(text: str) -> str:
    """Lowercase `text`, split punctuation into standalone tokens, collapse
    whitespace, and append the literal "<eos>" token.

    Returns a single space-separated token string (never empty: a passage with
    no content normalizes to just "<eos>").
    """
    normalized: str = PUNCTUATION_RE.sub(r" \1 ", text.lower())
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    if not normalized:
        return EOS_TOKEN
    return f"{normalized} {EOS_TOKEN}"


def build_vocab(
    passages: Iterable[str],
    vocab_size: int = VOCAB_SIZE,
    show_progress: bool = True,
) -> tuple[dict[str, int], list[str], Counter[str]]:
    """Count token frequencies over `passages` and build the id mappings.

    Reserved ids come first (0=<unk>, 1=<pad>, 2=<eos>); real tokens fill
    ids 3 .. vocab_size - 1 by descending frequency. Special-token strings found
    in the corpus are never duplicated into the real-token range.

    Returns (token_to_id, id_to_token, token_counts).
    """
    assert vocab_size <= 65535, (
        f"vocab_size {vocab_size} does not fit in uint16 (max 65535)"
    )
    counter: Counter[str] = Counter()
    iterator = (
        tqdm(passages, desc="counting", unit="passage", dynamic_ncols=True)
        if show_progress
        else passages
    )
    for passage in iterator:
        counter.update(normalize_passage(passage).split())

    real_budget: int = max(vocab_size - len(SPECIAL_TOKENS), 0)
    top_tokens: list[str] = [
        token for token, _count in counter.most_common() if token not in SPECIAL_TOKEN_SET
    ][:real_budget]

    id_to_token: list[str] = [*SPECIAL_TOKENS, *top_tokens]
    token_to_id: dict[str, int] = {token: idx for idx, token in enumerate(id_to_token)}

    assert len(id_to_token) <= 65535, (
        f"vocab_size {len(id_to_token)} does not fit in uint16 (max 65535)"
    )
    if len(id_to_token) != vocab_size:
        print(
            f"NOTE: corpus only had {len(top_tokens)} distinct real tokens; "
            f"vocab_size is {len(id_to_token)} instead of {vocab_size}."
        )

    return token_to_id, id_to_token, counter


def encode_passages(
    passages: Iterable[str],
    token_to_id: dict[str, int],
    unk_id: int = UNK_ID,
    desc: str = "encoding",
    show_progress: bool = True,
) -> list[int]:
    """Encode every passage into a flat list of token ids, using `unk_id` for
    tokens outside the vocabulary."""
    ids: list[int] = []
    iterator = (
        tqdm(passages, desc=desc, unit="passage", dynamic_ncols=True)
        if show_progress
        else passages
    )
    for passage in iterator:
        ids.extend(
            token_to_id.get(token, unk_id) for token in normalize_passage(passage).split()
        )
    return ids


def unknown_rate(ids: list[int], unk_id: int = UNK_ID) -> float:
    """Fraction of `ids` equal to `unk_id` (0.0 for an empty sequence)."""
    if not ids:
        return 0.0
    return sum(1 for token_id in ids if token_id == unk_id) / len(ids)


# ---------------------------------------------------------------------------
# 5. output
# ---------------------------------------------------------------------------


def write_vocab(path: Path, token_to_id: dict[str, int], id_to_token: list[str]) -> None:
    """Write {"token_to_id": ..., "id_to_token": ...} as JSON."""
    payload: dict[str, object] = {"token_to_id": token_to_id, "id_to_token": id_to_token}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ids(path: Path, ids: list[int], vocab_size: int) -> None:
    """Write a flat id stream as a headerless, little-endian uint16 .bin file."""
    assert vocab_size <= 65535, f"vocab_size {vocab_size} does not fit in uint16"
    assert not ids or max(ids) < vocab_size, "found a token id outside the vocabulary"
    np.asarray(ids, dtype="<u2").tofile(path)


def write_stats(path: Path, stats: dict[str, object]) -> None:
    """Write the run statistics as JSON."""
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def print_top_tokens(counter: Counter[str], limit: int = TOP_TOKENS_TO_SHOW) -> None:
    """Print the `limit` most frequent tokens, so the corpus can be eyeballed."""
    total: int = sum(counter.values())
    print(f"\ntop {limit} tokens")
    print("-" * 58)
    for rank, (token, count) in enumerate(counter.most_common(limit), start=1):
        share: float = count / total if total else 0.0
        print(f"  {rank:>2}. {token:<14} {count:>9,}  {share * 100:5.2f}%")


def summarize(
    stats: dict[str, object],
    train_passages: int,
    val_passages: int,
    written: list[Path],
) -> None:
    """Print the human-readable run summary, including the unknown-token rates
    as percentages and a WARNING when the training unknown rate exceeds 2%."""
    unk_train: float = float(stats["unk_rate_train"])
    unk_val: float = float(stats["unk_rate_val"])
    books: dict[str, int] = dict(stats["books"])  # type: ignore[arg-type]

    print("\n" + "=" * 58)
    print("prepare.py summary")
    print("=" * 58)
    print(f"source              {len(books)} gutenberg books in {RAW_DIR}")
    for name, count in books.items():
        print(f"  {name:<16}{count:>8,} passages")
    print(f"passages kept       {int(stats['stories_kept']):,}")
    print(f"passages dropped    {int(stats['stories_dropped']):,}  (< {MIN_PASSAGE_WORDS} words)")
    print(f"train passages      {train_passages:,}")
    print(f"val passages        {val_passages:,}  (every {VAL_EVERY}th, interleaved)")
    print(f"vocab size          {int(stats['vocab_size']):,}")
    print(f"train tokens        {int(stats['train_tokens']):,}")
    print(f"val tokens          {int(stats['val_tokens']):,}")
    print(f"unk_rate_train      {unk_train * 100:.3f}%")
    print(f"unk_rate_val        {unk_val * 100:.3f}%")
    for path in written:
        print(f"wrote               {path}")
    print("-" * 58)

    if unk_train > 0.02:
        print(f"WARNING: unk_rate_train {unk_train * 100:.3f}% exceeds 2.000%")

    print("=" * 58)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(
        description="Build a word-level corpus from local Gutenberg text files."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help=f"use whatever .txt files are already in {RAW_DIR} instead of fetching",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        metavar="N",
        help="cap the corpus at the first N passages, for quick test runs",
    )
    args = parser.parse_args(argv)
    if args.max_passages is not None and args.max_passages < 1:
        parser.error("--max-passages must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the full preparation pipeline and write all output files."""
    args = parse_args(argv)

    try:
        paths: list[Path] = (
            existing_books(RAW_DIR) if args.skip_download else download_books(RAW_DIR)
        )
        records, dropped = collect_passages(paths)
    except (RuntimeError, ValueError) as exc:
        sys.exit(str(exc))

    if args.max_passages is not None and len(records) > args.max_passages:
        print(f"\ncapping corpus at the first {args.max_passages:,} of {len(records):,} passages")
        records = records[: args.max_passages]

    passages: list[str] = [passage for _book, passage in records]
    per_book: Counter[str] = Counter(book for book, _passage in records)
    books: dict[str, int] = {
        book.name: per_book[book.name] for book in BOOKS if per_book[book.name]
    }

    if len(passages) < VAL_EVERY:
        sys.exit(
            f"only {len(passages)} passages built, need at least {VAL_EVERY} "
            "to hold out a validation split"
        )

    token_to_id, id_to_token, counter = build_vocab(passages)
    vocab_size: int = len(id_to_token)

    # Vocabulary is counted over the whole corpus; the holdout only happens
    # now, immediately before encoding. Order is preserved.
    train_passages: list[str] = [
        passage for index, passage in enumerate(passages) if index % VAL_EVERY != VAL_EVERY - 1
    ]
    val_passages: list[str] = [
        passage for index, passage in enumerate(passages) if index % VAL_EVERY == VAL_EVERY - 1
    ]

    print(f"\nencoding {len(train_passages):,} train passages and "
          f"{len(val_passages):,} val passages (vocab {vocab_size})")
    train_ids: list[int] = encode_passages(train_passages, token_to_id, desc="train")
    val_ids: list[int] = encode_passages(val_passages, token_to_id, desc="val")

    unk_rate_train: float = unknown_rate(train_ids)
    unk_rate_val: float = unknown_rate(val_ids)

    vocab_path: Path = OUTPUT_DIR / "vocab.json"
    train_path: Path = OUTPUT_DIR / "train.bin"
    val_path: Path = OUTPUT_DIR / "val.bin"
    stats_path: Path = OUTPUT_DIR / "stats.json"

    write_vocab(vocab_path, token_to_id, id_to_token)
    write_ids(train_path, train_ids, vocab_size)
    write_ids(val_path, val_ids, vocab_size)

    stats: dict[str, object] = {
        "vocab_size": vocab_size,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "unk_rate_train": unk_rate_train,
        "unk_rate_val": unk_rate_val,
        "stories_kept": len(passages),
        "stories_dropped": dropped,
        "books": books,
    }
    write_stats(stats_path, stats)

    print_top_tokens(counter)
    summarize(
        stats,
        train_passages=len(train_passages),
        val_passages=len(val_passages),
        written=[vocab_path, train_path, val_path, stats_path],
    )


if __name__ == "__main__":
    main()
