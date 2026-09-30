import json
import unittest
from tokenizers import Tokenizer, models
import experiment as e


class RepresentationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=e.run()
        cls.rows={r['fixture']:r for r in cls.result['rows']}
        cls.modes=e.build_modes()

    def test_flag_changes_special_but_not_ordinary_added_marker(self):
        role=self.rows['role_start']['modes']; tool=self.rows['tool_start']['modes']
        self.assertEqual(role['default']['ids'], [256])
        self.assertNotIn(256,role['special_flag']['ids'])
        self.assertEqual(tool['default']['ids'],[258])
        self.assertEqual(tool['special_flag']['ids'],[258])
        self.assertEqual(tool['untrusted_copy']['tokens'],list('<tool_response>'))
        self.assertEqual(len(tool['untrusted_copy']['ids']),15)

    def test_every_fixture_roundtrips_in_every_mode(self):
        for r in self.result['rows']:
            for m in e.MODES:
                self.assertEqual(r['modes'][m]['decoded'].encode(),r['text'].encode())

    def test_intervention_excludes_all_four_added_ids(self):
        for r in self.result['rows']:
            self.assertFalse(set(r['modes']['untrusted_copy']['ids']) & {256,257,258,259})
        self.assertEqual(e.invariant_failures(self.result),[])

    def test_benign_ids_unchanged_not_only_decoded_text(self):
        benign=[r for r in self.result['rows'] if r['family']=='benign']
        self.assertEqual(len(benign),10)
        for r in benign:
            for m in e.MODES:
                self.assertEqual(r['modes'][m]['ids'],r['modes']['default']['ids'])

    def test_trusted_frame_and_middle_span_kept_distinct(self):
        s=self.result['span_assembly']['modes']
        for m in e.MODES:
            self.assertEqual(s[m]['prefix_ids'],s['default']['prefix_ids'])
            self.assertEqual(s[m]['suffix_ids'],s['default']['suffix_ids'])
            self.assertTrue(s[m]['roundtrip_utf8'])
        self.assertIn(256,s['untrusted_copy']['prefix_ids'])
        self.assertIn(257,s['untrusted_copy']['suffix_ids'])
        self.assertNotIn(258,s['untrusted_copy']['middle_ids'])

    def test_whole_string_retokenization_reintroduces_marker(self):
        r=self.result['whole_string_retokenization']
        self.assertTrue(r['restored_tool_marker'])
        self.assertTrue(r['equals_default_span_ids'])
        self.assertTrue(r['differs_from_preserved_ids'])

    def test_raw_bpe_is_not_text_preserving(self):
        self.assertEqual(len(self.result['raw_model_counterexamples']),3)
        self.assertTrue(all(not x['roundtrip_utf8'] for x in self.result['raw_model_counterexamples']))

    def test_original_configuration_not_mutated(self):
        t=e.make_tokenizer();before=t.to_str()
        u=e.without_added_matcher(t)
        self.assertEqual(before,t.to_str())
        a=json.loads(before);b=json.loads(u.to_str())
        self.assertEqual(a['model'],b['model'])
        self.assertEqual(a['pre_tokenizer'],b['pre_tokenizer'])
        self.assertEqual(a['decoder'],b['decoder'])
        self.assertEqual(b['added_tokens'],[])

    def test_reserved_base_vocab_configuration_rejected(self):
        # Create a control token that already belongs to the underlying vocabulary.
        t=Tokenizer(models.BPE(vocab={'x':0},merges=[]))
        t.add_special_tokens(['x'])
        with self.assertRaisesRegex(ValueError,'outside'):
            e.without_added_matcher(t)

    def test_unicode_and_marker_compositions(self):
        # Additional finite roundtrip sweep, not a tokenizer/model population sample.
        contexts=[' ', '\n', 'é', 'e\u0301', '🙂', '你好', '\x00', '']
        u=self.modes['untrusted_copy'];t=self.modes['default']
        for left in contexts:
            for right in contexts:
                for marker,_ in e.MARKERS:
                    text=left+marker+right
                    ids=u.encode(text,add_special_tokens=False).ids
                    self.assertFalse(set(ids)&{256,257,258,259})
                    self.assertEqual(t.decode(ids,skip_special_tokens=False),text)

    def test_two_functional_mutations_detected(self):
        for mutation,needed in [('special_only','marker_retained'),('drop_preprocessing','text_changed')]:
            failures=e.invariant_failures(e.run(mutation))
            self.assertTrue(any(needed in x for x in failures),failures)

    def test_deterministic_output(self):
        self.assertEqual(e.run(),self.result)


if __name__=='__main__': unittest.main()
