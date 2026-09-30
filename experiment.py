"""Offline tokenizer representation assay. Synthetic vocabulary, real library, no model."""
import json
from pathlib import Path
import tokenizers
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, decoders

MARKERS = (("<|im_start|>", True), ("<|im_end|>", True),
           ("<tool_response>", False), ("</tool_response>", False))
FIXTURES = (
    ("role_start", "marker", "<|im_start|>"),
    ("role_end", "marker", "<|im_end|>"),
    ("tool_start", "marker", "<tool_response>"),
    ("tool_end", "marker", "</tool_response>"),
    ("mixed", "marker", "<|im_start|>note<tool_response>hello</tool_response><|im_end|>"),
    ("quoted_marker", "marker", 'Manual says "<tool_response>" is a label.'),
    ("adjacent", "marker", "<tool_response><tool_response>"),
    ("embedded", "marker", "a<tool_response>b"),
    ("ordinary", "benign", "The weather is sunny."),
    ("empty", "benign", ""),
    ("leading_spaces", "benign", "  hello  "),
    ("tabs_newline", "benign", "\talpha\nbeta\r\n"),
    ("composed", "benign", "café"),
    ("decomposed", "benign", "cafe\u0301"),
    ("emoji", "benign", "🙂 🧑🏽‍💻"),
    ("cjk", "benign", "你好，世界"),
    ("lookalike", "benign", "<tool_responsE>"),
    ("null_byte", "benign", "a\x00b"),
)
MODES = ("default", "special_flag", "untrusted_copy")


def make_tokenizer():
    # Complete byte alphabet, deliberately no merges or training. IDs are synthetic.
    vocab = {s: i for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    t = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    for text, special in MARKERS:
        t.add_tokens([AddedToken(text, special=special, normalized=False,
                                single_word=False, lstrip=False, rstrip=False)])
    return t


def without_added_matcher(original, mutation=None):
    config = json.loads(original.to_str())
    reserved = {a['id'] for a in config['added_tokens']}
    if config['model']['type'] != 'BPE' or reserved & set(config['model']['vocab'].values()):
        raise ValueError('Unsupported: control IDs must be outside a BPE base vocabulary')
    config['added_tokens'] = ([a for a in config['added_tokens'] if not a['special']]
                              if mutation == 'special_only' else [])
    if mutation == 'drop_preprocessing':
        config['pre_tokenizer'] = None
    return Tokenizer.from_str(json.dumps(config))


def build_modes(mutation=None):
    original = make_tokenizer()
    flag = Tokenizer.from_str(original.to_str())
    flag.encode_special_tokens = True
    return {'default': original, 'special_flag': flag,
            'untrusted_copy': without_added_matcher(original, mutation)}


def observe(t, text, reserved):
    encoded = t.encode(text, add_special_tokens=False)
    decoded = t.decode(encoded.ids, skip_special_tokens=False)
    return {'ids': encoded.ids, 'tokens': encoded.tokens, 'decoded': decoded,
            'roundtrip_utf8': decoded.encode('utf-8') == text.encode('utf-8'),
            'marker_ids': [i for i in encoded.ids if i in reserved]}


def run(mutation=None):
    modes = build_modes(mutation)
    original = modes['default']
    config = json.loads(original.to_str())
    reserved = {a['id'] for a in config['added_tokens']}
    rows = [{'fixture': name, 'family': family, 'text': text,
             'modes': {name: observe(t, text, reserved) for name, t in modes.items()}}
            for name, family, text in FIXTURES]
    prefix, untrusted, suffix = '<|im_start|>user\n', '<tool_response>', '\n<|im_end|>'
    prefix_ids = original.encode(prefix, add_special_tokens=False).ids
    suffix_ids = original.encode(suffix, add_special_tokens=False).ids
    span_rows = {}
    for name, t in modes.items():
        middle_ids = t.encode(untrusted, add_special_tokens=False).ids
        assembled = prefix_ids + middle_ids + suffix_ids
        decoded = original.decode(assembled, skip_special_tokens=False)
        span_rows[name] = {'prefix_ids': prefix_ids, 'middle_ids': middle_ids,
                          'suffix_ids': suffix_ids, 'assembled_ids': assembled,
                          'decoded': decoded, 'roundtrip_utf8': decoded == prefix+untrusted+suffix}
    decoded = span_rows['untrusted_copy']['decoded']
    reencoded = original.encode(decoded, add_special_tokens=False).ids
    raw_examples = []
    for text in ['  hello', 'cafe\u0301', '🙂']:
        try:
            ids = [t.id for t in original.model.tokenize(text)]
            raw_decoded = original.decode(ids, skip_special_tokens=False)
            raw_examples.append({'text': text, 'ids': ids, 'decoded': raw_decoded,
                                 'roundtrip_utf8': raw_decoded == text})
        except Exception as exc:
            raw_examples.append({'text': text, 'error_type': type(exc).__name__,
                                 'roundtrip_utf8': False})
    return {'schema_version': 1, 'tokenizers_version': tokenizers.__version__,
            'scope': 'Real tokenizer library, synthetic byte vocabulary; no model or action',
            'construction': {'base_vocab_size': 256, 'merges': [],
                             'added_tokens': config['added_tokens'], 'synthetic_ids': True},
            'fixture_count': len(rows), 'mode_count': len(MODES), 'rows': rows,
            'counts': {m: {'roundtrip_rows': sum(r['modes'][m]['roundtrip_utf8'] for r in rows),
                          'marker_bearing_rows': sum(bool(r['modes'][m]['marker_ids']) for r in rows),
                          'benign_ids_preserved': sum(r['family']=='benign' and r['modes'][m]['ids']==r['modes']['default']['ids'] for r in rows)} for m in MODES},
            'span_assembly': {'prefix': prefix, 'untrusted': untrusted, 'suffix': suffix,
                              'modes': span_rows},
            'whole_string_retokenization': {'ids': reencoded,
                'equals_default_span_ids': reencoded == span_rows['default']['assembled_ids'],
                'differs_from_preserved_ids': reencoded != span_rows['untrusted_copy']['assembled_ids'],
                'restored_tool_marker': original.token_to_id('<tool_response>') in reencoded},
            'raw_model_counterexamples': raw_examples}


def invariant_failures(result):
    failures = []
    for row in result['rows']:
        obs = row['modes']['untrusted_copy']
        if obs['marker_ids']: failures.append(row['fixture'] + ':marker_retained')
        if not obs['roundtrip_utf8']: failures.append(row['fixture'] + ':text_changed')
        if row['family']=='benign' and obs['ids'] != row['modes']['default']['ids']:
            failures.append(row['fixture'] + ':benign_ids_changed')
    return failures


def write_evidence(result):
    lines = ['# Exact tokenizer evidence', '',
             'Actual tokenizers '+result['tokenizers_version']+'; synthetic byte vocabulary and IDs. No model, agent or action.', '',
             '18 chosen inputs × 3 modes = 54 rows. Counts are fixture observations, not attack rates.', '',
             '| Mode | Exact text roundtrips /18 | Rows containing added marker IDs /18 | Benign IDs preserved /10 |',
             '|---|---:|---:|---:|']
    for mode, c in result['counts'].items():
        lines.append(f"| {mode} | {c['roundtrip_rows']} | {c['marker_bearing_rows']} | {c['benign_ids_preserved']} |")
    lines += ['', '## Single-marker traces', '', '| Input | Default IDs | Special flag IDs | Untrusted-copy IDs |', '|---|---|---|---|']
    for r in result['rows'][:4]:
        cells = [r['fixture']] + [str(r['modes'][m]['ids']) for m in MODES]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'Both encodings decode to the exact same input text. Marker presence is a representation oracle, not observed model authority.', '',
              '## Assembly and counterexamples', '',
              'Trusted prefix/suffix use the default tokenizer. Only the middle result span uses the copy. Full traces are in artifacts/run.json.', '',
              'Decoding the assembled intervention IDs and normally re-tokenizing the whole string restores the tool marker ID. Keep the chosen IDs through the model boundary; a string-only interface cannot preserve this distinction.', '',
              'Calling raw BPE without its byte preprocessor loses text in all three chosen whitespace/accent/emoji examples. This shortcut is not equivalent to removing added matcher entries.', '',
              'Unsupported control IDs in the base vocabulary are rejected by construction; this lab does not implement a constrained encoder for them.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    result = run()
    if invariant_failures(result): raise SystemExit(invariant_failures(result))
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/run.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    Path('evidence.md').write_text(write_evidence(result))
    print(json.dumps(result['counts']))
