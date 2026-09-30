# Exact tokenizer evidence

Actual tokenizers 0.23.1; synthetic byte vocabulary and IDs. No model, agent or action.

18 chosen inputs × 3 modes = 54 rows. Counts are fixture observations, not attack rates.

| Mode | Exact text roundtrips /18 | Rows containing added marker IDs /18 | Benign IDs preserved /10 |
|---|---:|---:|---:|
| default | 18 | 8 | 10 |
| special_flag | 18 | 6 | 10 |
| untrusted_copy | 18 | 0 | 10 |

## Single-marker traces

| Input | Default IDs | Special flag IDs | Untrusted-copy IDs |
|---|---|---|---|
| role_start | [256] | [27, 91, 72, 76, 62, 82, 83, 64, 81, 83, 91, 29] | [27, 91, 72, 76, 62, 82, 83, 64, 81, 83, 91, 29] |
| role_end | [257] | [27, 91, 72, 76, 62, 68, 77, 67, 91, 29] | [27, 91, 72, 76, 62, 68, 77, 67, 91, 29] |
| tool_start | [258] | [258] | [27, 83, 78, 78, 75, 62, 81, 68, 82, 79, 78, 77, 82, 68, 29] |
| tool_end | [259] | [259] | [27, 14, 83, 78, 78, 75, 62, 81, 68, 82, 79, 78, 77, 82, 68, 29] |

Both encodings decode to the exact same input text. Marker presence is a representation oracle, not observed model authority.

## Assembly and counterexamples

Trusted prefix/suffix use the default tokenizer. Only the middle result span uses the copy. Full traces are in artifacts/run.json.

Decoding the assembled intervention IDs and normally re-tokenizing the whole string restores the tool marker ID. Keep the chosen IDs through the model boundary; a string-only interface cannot preserve this distinction.

Calling raw BPE without its byte preprocessor loses text in all three chosen whitespace/accent/emoji examples. This shortcut is not equivalent to removing added matcher entries.

Unsupported control IDs in the base vocabulary are rejected by construction; this lab does not implement a constrained encoder for them.
