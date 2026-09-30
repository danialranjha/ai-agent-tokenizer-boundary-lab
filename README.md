# AI-agent tokenizer boundary lab

A tokenizer configuration assay using **Hugging Face tokenizers 0.23.1 and a synthetic byte-level vocabulary**. No model weights, inference, agent, shell action, external endpoint or malicious instruction is involved. Token IDs in this repository are synthetic, not Qwen or DeepSeek IDs.

## Hypothesis

A flag that splits special tokens can leave ordinary added protocol markers atomic. Removing added-token matcher entries for an untrusted span, while keeping byte preprocessing and decoding, can remove those marker IDs without changing the decoded text. The same change must not alter trusted framing.

Motivation: [Same Bytes, Different Authority, Appendix A.3/G.1](https://arxiv.org/html/2609.35932v1) distinguishes marker matching from ordinary text processing. This lab independently tests that configuration distinction; it does not reproduce the paper's model results.

## Setup and exact commands

Python 3.11+ and the pinned dependency. Installation requires package access; the experiment itself is offline.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiment.py
.venv/bin/python -m unittest -v test_experiment
.venv/bin/python run_checks.py
```

The reference run used Python 3.11.6 and an existing cached installation of tokenizers 0.23.1 in an isolated environment. No network install was performed for that run. Commands regenerate [artifacts/run.json](artifacts/run.json), [evidence.md](evidence.md) and [artifacts/checks.json](artifacts/checks.json). With the pinned version, generated evidence is deterministic; timestamps and host paths are omitted.

## Threat model

An untrusted source can choose the text of one result span, including marker-shaped strings. The host, tokenizer configuration, trust-span labels and ID assembly remain trusted. Fixtures contain harmless labels and ordinary text. We measure only the representation that could later be passed to a self-hosted model. We do not measure an attacker persuading a model, an action being authorized, or an exploit succeeding.

The synthetic vocabulary contains the full 256-byte alphabet, sorted for stable IDs, and no BPE merges. BPE means byte-pair encoding, a way of combining text units; no learned combinations are needed for this matcher assay. The real library's ByteLevel preprocessing and decoder remain active. Four added markers are assigned IDs256–259: chat-start/end are special, tool-response start/end are non-special. Flags, IDs and the exact construction are saved in run.json.

## Technical walkthrough

Three configurations process the same 18 selected inputs:

- **Default matching:** Every added marker retains its single synthetic ID.
- **Special-token flag:** Set the library's native `encode_special_tokens=True`; special markers split, ordinary added markers remain matched.
- **Untrusted-span copy:** Remove all added matcher entries from a cloned tokenizer configuration; preserve its base vocabulary, normalizer, preprocessor and decoder.

For this supported construction, all control IDs are outside the base vocabulary. A guard refuses overlapping IDs or a non-BPE configuration. It is not a general tokenizer validator and does not implement constrained encoding for base-vocabulary control IDs.

### Results

| Configuration | Exact text roundtrips /18 | Inputs retaining marker IDs /18 | Marker-free benign IDs unchanged /10 |
|---|---:|---:|---:|
| Default matching | 18 | 8 | 10 |
| Special-token flag | 18 | 6 | 10 |
| Untrusted-span copy | 18 | 0 | 10 |

For the exact text `<tool_response>`, Default matching and Special-token flag return `[258]`. Untrusted-span copy emits 15 ordinary byte-token IDs. All three decode to the identical text. For `<|im_start|>`, the special flag does split synthetic ID256. This positive control proves that the flag was actually active.

The trusted prefix and suffix retain their original IDs while only the middle result span changes. Decoding that assembled sequence and re-encoding the whole string with Default matching restores the tool marker ID. A later string-only interface therefore undoes the selected representation. These are exact observations, not inferred model outcomes.

A separate wrong shortcut calls raw BPE without ByteLevel preprocessing. It loses text in three selected leading-space, decomposed-accent and emoji cases. Preserving the full text-processing pipeline matters.

## Validation

Twelve tests pass, including 256 combinations of marker and Unicode/whitespace context. They check exact bytes and IDs, active flag behavior, preserved trusted framing, ordinary inputs, unsupported configuration rejection and re-tokenization. Functional negative variants retain non-special matchers or drop preprocessing; the invariant oracle detects both. Counts and failing invariants are in checks.json. These mutations test the oracle, not the security of arbitrary production configurations.

## Limits and engineering check

This vocabulary is intentionally synthetic and has no learned merges or model. The experiment does not establish model authority, attack success, Qwen tokenization, real traffic prevalence, general Unicode coverage or production serving behavior. Quoted marker text is harmless but still representation-sensitive; changing its encoding is not a maliciousness judgment. Ordinary words remain intact and may still contain unwanted instructions.

Trust labels and the complete downstream ID path are assumed correct. Real deployments need their actual tokenizer, normalization, postprocessor, chat template, truncation and serving boundary tested. Removing matcher entries alone is insufficient where control IDs can be emitted by the base vocabulary. Valid UTF-8 fixtures are used; invalid Unicode inputs are outside scope.

For this byte-level construction, test both special and non-special added markers in untrusted spans, preserve ordinary preprocessing, and carry the selected IDs forward without decoding and normally re-tokenizing them. Pair marker exclusion checks with exact text roundtrips and unchanged trusted framing before drawing any model-level conclusion.
