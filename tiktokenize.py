from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import json


tokenizer_path = "./content/tokenizer.model"
num_reserved_special_tokens = 256


mergeable_ranks = load_tiktoken_bpe(tokenizer_path)


num_base_tokens = len(mergeable_ranks)
special_tokens = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
    "<|eot_id|>",
    "<|python_tag|>",
]

reserved_tokens = [
    f"<|reserved_special_token_{2 + i}|>" for i in range(num_reserved_special_tokens - len(special_tokens))
]
special_tokens = special_tokens + reserved_tokens


tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^rnp{L}p{N}]?p{L}+|p{N}{1,3}| ?[^sp{L}p{N}]+[rn]*|s*[rn]+|s+(?!S)|s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)
