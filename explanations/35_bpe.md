This class implements a small **Byte Pair Encoding (BPE) tokenizer**. It learns which adjacent characters or character groups frequently occur together, then treats those groups as single tokens.

Despite the name “Byte Pair Encoding,” this implementation starts with **Python characters**, not raw bytes.

## 1. The basic idea

Suppose a word initially looks like this:

```text
lower → ['l', 'o', 'w', 'e', 'r', '</w>']
```

BPE repeatedly merges common neighboring symbols:

```text
'l' + 'o'   → 'lo'
'lo' + 'w'  → 'low'
```

The word can then be represented as:

```text
['low', 'e', 'r', '</w>']
```

The special symbol `'</w>'` marks the **end of a word**. It can also become part of a merged token, such as `'low</w>'`.

The class has two main operations:

- **`train()`**: learn which pairs to merge, and in what order.
- **`encode()`**: apply those learned merges to new text.

---

## 2. Initialization

```python
def __init__(self):
    self.merges = []
```

`self.merges` stores the learned merge rules in order.

For example:

```python
[
    ('l', 'o'),
    ('lo', 'w'),
    ('low', '</w>')
]
```

These rules mean:

1. Merge adjacent `'l'` and `'o'` into `'lo'`.
2. Merge adjacent `'lo'` and `'w'` into `'low'`.
3. Merge adjacent `'low'` and `'</w>'` into `'low</w>'`.

**The order matters**, because a later rule may depend on a token created by an earlier rule.

---

## 3. Training: `train(corpus, num_merges)`

Here, `corpus` is expected to be an iterable of **individual words**, such as:

```python
corpus = ['low', 'low', 'lower']
```

`num_merges` is the maximum number of merge rules to learn.

### Step A: Count words in their initial character form

```python
vocab = {}
for word in corpus:
    symbols = tuple(word) + ('</w>',)
    vocab[symbols] = vocab.get(symbols, 0) + 1
```

Each word becomes a tuple of characters followed by the end-of-word marker.

For the example above, `vocab` becomes:

```python
{
    ('l', 'o', 'w', '</w>'): 2,
    ('l', 'o', 'w', 'e', 'r', '</w>'): 1
}
```

The numbers are word frequencies.

Tuples are used because they can serve as dictionary keys; lists cannot.

### Step B: Count every adjacent pair

```python
pairs = {}
for word, freq in vocab.items():
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        pairs[pair] = pairs.get(pair, 0) + freq
```

For each word, the code examines neighboring symbols.

In our example:

| Pair | Total count |
|---|---:|
| `('l', 'o')` | 3 |
| `('o', 'w')` | 3 |
| `('w', '</w>')` | 2 |
| `('w', 'e')` | 1 |
| `('e', 'r')` | 1 |
| `('r', '</w>')` | 1 |

The counts are **weighted by word frequency**. Because `'low'` appears twice, each pair inside it contributes twice.

### Step C: Choose the most frequent pair

```python
if not pairs:
    break
best = max(pairs, key=pairs.get)
self.merges.append(best)
```

If there are no adjacent pairs left, training stops.

Otherwise, the most frequent pair is selected and recorded.

Here, `('l', 'o')` and `('o', 'w')` tie. Python’s `max()` selects the first encountered maximum, so this example chooses:

```python
('l', 'o')
```

### Step D: Merge that pair throughout the vocabulary

```python
new_vocab = {}
for word, freq in vocab.items():
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == best:
            new_word.append(word[i] + word[i + 1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    new_vocab[tuple(new_word)] = freq
vocab = new_vocab
```

The code scans each word from left to right:

- If the next two symbols match the chosen pair, concatenate them and advance by **two** positions.
- Otherwise, copy the current symbol and advance by **one** position.

After merging `('l', 'o')`:

```text
('l', 'o', 'w', '</w>')
→ ('lo', 'w', '</w>')

('l', 'o', 'w', 'e', 'r', '</w>')
→ ('lo', 'w', 'e', 'r', '</w>')
```

It then repeats the pair-counting and merging process, using the **updated** vocabulary.

With three merges, our example learns:

```python
[
    ('l', 'o'),
    ('lo', 'w'),
    ('low', '</w>')
]
```

Calling `train()` again replaces the previous rules because it resets `self.merges`.

---

## 4. Encoding: `encode(text)`

Encoding uses the learned rules—it does **not** calculate new pair frequencies.

### Step A: Split the text into words

```python
all_tokens = []
for word in text.split():
    symbols = list(word) + ['</w>']
```

For example:

```text
"low lower"
```

is processed as two words:

```python
['l', 'o', 'w', '</w>']
['l', 'o', 'w', 'e', 'r', '</w>']
```

### Step B: Apply each merge rule in training order

```python
for a, b in self.merges:
    i = 0
    while i < len(symbols) - 1:
        if symbols[i] == a and symbols[i + 1] == b:
            symbols = symbols[:i] + [a + b] + symbols[i + 2:]
        else:
            i += 1
```

This expression:

```python
symbols[:i] + [a + b] + symbols[i + 2:]
```

rebuilds the list as:

```text
symbols before the pair + merged token + symbols after the pair
```

For `'low'`, the learned rules produce:

```text
['l', 'o', 'w', '</w>']
→ ['lo', 'w', '</w>']
→ ['low', '</w>']
→ ['low</w>']
```

For `'lower'`:

```text
['l', 'o', 'w', 'e', 'r', '</w>']
→ ['lo', 'w', 'e', 'r', '</w>']
→ ['low', 'e', 'r', '</w>']
```

The last rule, `('low', '</w>')`, does not apply to `'lower'`: `'low'` is followed by `'e'`, not the end-of-word marker.

### Step C: Collect the tokens

```python
all_tokens.extend(symbols)
return all_tokens
```

Tokens from every word are added to one flat list.

---

## 5. Complete example

```python
bpe = SimpleBPE()

bpe.train(['low', 'low', 'lower'], num_merges=3)

print(bpe.merges)
# [('l', 'o'), ('lo', 'w'), ('low', '</w>')]

print(bpe.encode('low lower'))
# ['low</w>', 'low', 'e', 'r', '</w>']
```

## Important limitations

- **Training expects words.** Passing a raw string directly to `train()` would iterate over its characters, not its words.
- **Whitespace is not preserved.** `text.split()` removes leading/trailing whitespace and collapses separators.
- **No merges cross word boundaries.** Each word is processed separately.
- **Output tokens are strings, not integer token IDs.**
- **Unseen words still work:** their characters are merged wherever existing rules apply; the remaining characters stay separate.
- **This is an educational implementation:** it lacks the efficiency and additional features of production tokenizers.

**In short:** training learns frequent adjacent-symbol combinations; encoding reuses those combinations to turn text into character or subword tokens.