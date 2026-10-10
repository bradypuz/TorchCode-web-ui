# Byte-Pair Encoding

## Intuition

BPE builds reusable subword pieces by repeatedly combining the most frequent neighboring symbols. Starting from characters makes unseen words representable, while repeated merges compress common spellings into fewer tokens. This exercise implements a small word-based tokenizer with an explicit end-of-word marker.

## Inputs and output

`train(corpus, num_merges)` receives a list of word strings and stores an ordered list of symbol pairs in `self.merges`. It returns nothing. `encode(text)` splits text on whitespace and returns a flat `list[str]` of tokens. There are no tensor dimensions or integer vocabulary IDs; tokens can include the `</w>` marker.

## Implementation walkthrough

1. Represent each training word as its character tuple followed by `</w>`. Count repeated words in a vocabulary dictionary.
2. Clear previous merge rules so each `train()` call starts fresh.
3. Count adjacent symbol pairs across the current vocabulary. Weight every occurrence by the word's corpus frequency, including repeated occurrences within a word.
4. Select the pair with the greatest count and record it. If no adjacent pairs remain, stop before reaching the requested merge count.
5. Rebuild each word's symbol sequence, merging matching pairs left to right and advancing two positions after each merge to avoid overlapping replacements.
6. To encode new text, start each whitespace-separated word at character level and replay every learned merge in training order. Concatenate all resulting word tokens.

## Why it works

The score for pair $(a,b)$ is its weighted occurrence count:

$$
C(a,b)=\sum_w f(w)\sum_i\mathbf{1}\{(s_{w,i},s_{w,i+1})=(a,b)\}
$$

Replacing the largest-count pair locally reduces the number of symbols used by common patterns. Later merges depend on earlier merged symbols, so rule order matters. The end marker lets a frequent word ending be represented distinctly. The judge checks merge count, the first frequent pair, reconstruction, and the tendency for more merges to produce fewer tokens.

## Pitfalls and reference scope

Do not count every distinct word equally; repeated words must affect pair frequency. Do not apply all rules simultaneously during encoding. Equal-frequency pairs are resolved by dictionary insertion order through Python's `max`, so corpus order can affect ties. Despite the name, this reference starts with Unicode characters, not raw bytes. It does not preserve original whitespace, supply decoding or token IDs, use a pre-tokenization regex, or add special-token handling. The end marker may remain separate or become part of a merged token.
