import regex as re
from .common import FIXTURES_PATH
from cs336_basics.pretokenization import (
    PAT,pretokenize_count, split_tokens,
    create_merge_pair_counts,find_max_merge_pair,
    merge_bpe
)

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
bpe_example = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

EXPECTED_TOKENS = {"low": 5, "lower": 2, "widest": 3, "newest": 6}


def test_pretokenize_count():
    tokens = pretokenize_count(bpe_example, pattern=r"\w+")
    assert tokens == EXPECTED_TOKENS

def test_pat():
    text = "some text that i'll pre-tokenize"
    tokens = re.findall(PAT, text)
    assert tokens == ["some", " text", " that", " i", "'ll", " pre", "-", "tokenize"]

def test_split_tokens():
    token_counts = {"low":5, "high":2}
    expected = {("l","o","w"):5,
                ("h","i","g","h"):2}
    actual = split_tokens(token_counts)
    assert expected==actual


def test_merge_byte_pair_counts():
    byte_tuple_counts = split_tokens(EXPECTED_TOKENS)
    byte_pair_counts, byte_pair_locations = create_merge_pair_counts(byte_tuple_counts)
    breakpoint()

def test_find_max_merge_pairs():
    byte_pair_counts = {
        "lo": 7, "ow": 7, "we": 8, "er": 2, "wi": 3, 
        "id": 3, "de": 3, "es": 9, "st": 9, "ne": 6, 
        "ew": 6}
    max_byte_pair = find_max_merge_pair(byte_pair_counts)
    expected ="st"
    assert max_byte_pair == expected

def test_merge_bpe():
    bpe_counts = EXPECTED_TOKENS
    bpe_counts = split_tokens(bpe_counts)
    bpe_counts,merge_pairs = merge_bpe(bpe_counts,6)
    assert merge_pairs == ['st', 'est', 'ow', 'low', 'west', 'ne']

