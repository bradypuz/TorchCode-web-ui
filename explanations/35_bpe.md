# Byte-Pair Encoding

This tokenizer learns reusable word pieces by repeatedly merging the most frequent neighboring symbols. It starts with characters and an end-of-word marker, so a word can still be represented even when the complete word was absent from training.

## Inputs and output

`SimpleBPE.train(corpus, num_merges)` takes a list of word strings and stores ordered pair rules in `self.merges`; it returns nothing. `encode(text)` splits a string on whitespace and returns a flat list of string tokens. There are no tensors or integer token IDs. A token can contain the literal `</w>` marker.

## Reading the actual code

```python
vocab = {}
for word in corpus:
    symbols = tuple(word) + ('</w>',)
    vocab[symbols] = vocab.get(symbols, 0) + 1
self.merges = []
```

A word such as `ab` becomes `('a', 'b', '</w>')`. Tuples can be dictionary keys. Repeated words increment their frequency instead of creating separate entries. The end marker identifies word endings and can itself be merged into a token. Resetting `self.merges` makes each training call start a new tokenizer.

```python
pairs = {}
for word, freq in vocab.items():
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        pairs[pair] = pairs.get(pair, 0) + freq
```

Every adjacent position contributes its word frequency. With corpus `['ab', 'ab', 'ac']`, the vocabulary has frequency two for `ab` and one for `ac`. Pair counts are:

| Adjacent pair | Count from `ab` | Count from `ac` | Total |
| --- | ---: | ---: | ---: |
| `('a','b')` | 2 | 0 | 2 |
| `('b','</w>')` | 2 | 0 | 2 |
| `('a','c')` | 0 | 1 | 1 |
| `('c','</w>')` | 0 | 1 | 1 |

Repeated occurrences within one word count separately too: `aaaa` contains three adjacent `('a','a')` occurrences before any merge. Pair counting considers overlapping positions; performing a merge uses nonoverlapping replacements.

```python
if not pairs:
    break
best = max(pairs, key=pairs.get)
self.merges.append(best)
```

`max` chooses the greatest count. Equal counts keep the first encountered dictionary key, so `('a','b')` wins our first tie. Recording rules in order matters because a later rule can use a symbol created by an earlier one. With no adjacent pairs left, training stops early.

```python
while i < len(word):
    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
        new_word.append(word[i] + word[i + 1])
        i += 2
    else:
        new_word.append(word[i])
        i += 1
new_vocab[tuple(new_word)] = freq
```

The scan builds a replacement word left to right. Concatenating two symbols creates one new symbol, and advancing by two prevents reusing the second symbol in an overlapping merge. Unmatched symbols are copied. The outer loop repeats pair counting on this rebuilt vocabulary.

```text
ab, frequency 2:  [a, b, </w>] -> [ab, </w>] -> [ab</w>]
ac, frequency 1:  [a, c, </w>] -> [a, c, </w>] -> [a, c, </w>]
merge rules:      ('a','b')       ('ab','</w>')
```

After the first merge, `('ab','</w>')` has count two and wins the second round.

```python
for word in text.split():
    symbols = list(word) + ['</w>']
    for a, b in self.merges:
```

Encoding restarts at characters for each whitespace-separated word, then replays every learned pair in order. Its scan replaces a match with `symbols[:i] + [a+b] + symbols[i+2:]`; unmatched positions advance `i`. Finally, `all_tokens.extend(symbols)` joins the words' token lists. Applying rules simultaneously would miss later rules involving `ab`.

## Run two merges

Load the reference solution first, then run:

```python
# Example: Count repeated words and replay learned merges in order
bpe = SimpleBPE()
bpe.train(['ab', 'ab', 'ac'], num_merges=2)
assert bpe.merges == [('a', 'b'), ('ab', '</w>')]
assert bpe.encode('ab ac') == ['ab</w>', 'a', 'c', '</w>']
assert bpe.encode('ad') == ['a', 'd', '</w>']
overlap = SimpleBPE()
overlap.train(['aaaa'], num_merges=1)
assert overlap.merges == [('a', 'a')]
assert overlap.encode('aaaa') == ['aa', 'aa', '</w>']
```

## Common mistakes and reference limits

Count corpus frequency, preserve merge order, and avoid overlapping replacements. Corpus order influences ties. Despite the name, this reference starts from Unicode characters rather than bytes. It does not preserve whitespace, implement decoding, assign vocabulary IDs, or add special-token handling. The marker may stay separate or become part of a larger token.
