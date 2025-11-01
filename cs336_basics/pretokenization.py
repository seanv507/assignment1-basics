import os
from typing import BinaryIO, Iterable
from collections import Counter, defaultdict
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
def chunk_and_count_tokens(filename, special_tokens=None):
    with open(filename, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        token_counts = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            token_count = process_chunk(f, start, end, special_tokens)
            token_counts += token_count
    byte_tuple_counts = split_tokens(token_counts)
    print(byte_tuple_counts)
    byte_pair_counts, byte_pair_locations = create_byte_pair_counts(byte_tuple_counts)
    sorted_byte_pairs = sort_byte_pairs(byte_pair_counts)
    print(sorted_byte_pairs)
    return byte_tuple_counts


def split_tokens(token_counts: dict[str, int]) -> dict[tuple[bytes], int]:
    byte_tuple_counts = Counter({tuple(token): count for token, count in token_counts.items()})
    return byte_tuple_counts


def process_chunk(file: BinaryIO, start: int, end: int, special_tokens):
    text = read_chunk(file, start, end)

    texts = strip_special_tokens(text, special_tokens)
    token_counts = count_tokens(tok for text in texts for tok in pretokenize(text))
    return token_counts


def read_chunk(file: BinaryIO, start: int, end: int) -> str:
    file.seek(start)
    chunk = file.read(end - start).decode("utf-8", errors="ignore")
    return chunk


def strip_special_tokens(text: str, special_tokens: Iterable[str] = None) -> list[str]:
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    regex_split = "|".join(escaped_special_tokens)
    texts = re.split(regex_split, text)
    return texts


def pretokenize(text: str) -> dict[str, int]:
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    tokens = re.finditer(PAT, text)
    for token in tokens:
        yield token.group()


def count_tokens(tokens: Iterable[str]) -> dict[str, int]:
    token_counts = Counter(tokens)

    return token_counts


def create_byte_pair_counts(byte_tuple_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    byte_pair_counts = Counter()
    byte_pair_locations = defaultdict(set)
    for byte_tuple, count in byte_tuple_counts.items():
        if len(byte_tuple) == 1:
            continue
        for b1, b2 in zip(byte_tuple, byte_tuple[1:]):
            byte_pair_counts[(b1, b2)] += count
            byte_pair_locations[(b1, b2)].add(byte_tuple)
    return byte_pair_counts, byte_pair_locations


def sort_byte_pairs(byte_pair_counts: dict[tuple[bytes], int]) -> list[tuple[bytes]]:
    sorted_byte_pairs = sorted(byte_pair_counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_byte_pairs


def merge_byte_pairs(byte_pair_counts, byte_pair_locations, byte_tuple_counts, pair_to_merge):
    b1, b2 = pair_to_merge
    new_byte = b1 + b2
    new_byte_tuple_counts = Counter()
    for byte_tuple, count in byte_tuple_counts.items():
        if byte_tuple in byte_pair_locations[pair_to_merge]:
            new_tuple = []
            skip_next = False
            for b in byte_tuple:
                if skip_next:
                    skip_next = False
                    continue
                if b == b1:
                    next_index = byte_tuple.index(b) + 1
                    if next_index < len(byte_tuple) and byte_tuple[next_index] == b2:
                        new_tuple.append(new_byte)
                        skip_next = True
                    else:
                        new_tuple.append(b)
                else:
                    new_tuple.append(b)
            new_byte_tuple_counts[tuple(new_tuple)] += count
        else:
            new_byte_tuple_counts[byte_tuple] += count
    return new_byte_tuple_counts


#     #
# sort by count

# # how to merge byte pairs (rather than chars?), tuples?
# how to do reversed lexicographic ordering on tuples of different nesting?
#
# sort counters by count and lex ordering
# then go through and replace key with merged version in byte_tuple_counts

# repeat again
