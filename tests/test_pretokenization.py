import regex as re
from typing import BinaryIO, Iterable
from .common import FIXTURES_PATH
from cs336_basics.pretokenization import chunk_and_count_tokens, sort_byte_pairs
from cs336_basics.pretokenization import PAT

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def test_chunk():
    corpus_path = FIXTURES_PATH / "bpe_example.txt"
    tokens = chunk_and_count_tokens(corpus_path, special_tokens=None)


def test_pat():
    text = "some text that i'll pre-tokenize"
    tokens = re.findall(PAT, text)
    assert tokens == ["some", " text", " that", " i", "'ll", " pre", "-", "tokenize"]


def test_sorted_byte_pairs():
    byte_pair_counts = {(b"a", b"b"): 7, (b"b", b"c"): 7, (b"c", b"d"): 3}

    sorted_pairs = sort_byte_pairs(byte_pair_counts)
    # wrong ordering for now
    expected_sorted_pairs = [
        ((b"a", b"b"), 7),  # 7 from (b,c)
        ((b"b", b"c"), 7),  # lexicographic ordering
        ((b"c", b"d"), 3),  # 3 from (c,d)
    ]
    assert sorted_pairs == expected_sorted_pairs
