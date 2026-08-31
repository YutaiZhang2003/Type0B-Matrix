import json
from pathlib import Path
import tempfile
import unittest

from fivepoint_sample_journal import SampleJournal


class SampleJournalTests(unittest.TestCase):
    def test_exception_flushes_and_resume_reuses_exact_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'samples.jsonl'
            with self.assertRaises(RuntimeError):
                with SampleJournal(p,'s') as journal:
                    for i in range(37):
                        journal.evaluate(str(i),lambda i=i: i+2j*i)
                    raise RuntimeError('interrupt')
            with SampleJournal(p,'s') as restored:
                for i in range(37):
                    self.assertEqual(restored.evaluate(str(i),lambda: self.fail('recomputed')),i+2j*i)
                self.assertEqual(restored.reused,37)
                with self.assertRaises(BlockingIOError):
                    SampleJournal(p,'s')
            with self.assertRaisesRegex(ValueError,'signature mismatch'):
                SampleJournal(p,'other')

    def test_incomplete_tail_is_recomputed_but_complete_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'samples.jsonl'
            with SampleJournal(p,'s') as journal:
                journal.evaluate('good',lambda:3+4j)
            with p.open('ab') as handle:
                handle.write(b'{"key":"interrupted"')
            with SampleJournal(p,'s') as journal:
                self.assertEqual(journal.evaluate('good',lambda:0),3+4j)
                journal.evaluate('interrupted',lambda:5+6j)
                with self.assertRaises(ArithmeticError):
                    journal.evaluate('bad',lambda:complex('nan'))
            with SampleJournal(p,'s') as journal:
                self.assertEqual(len(journal.values),2)
            with p.open('ab') as handle:
                handle.write(b'invalid complete record\n')
            with self.assertRaises(json.JSONDecodeError):
                SampleJournal(p,'s')


if __name__ == '__main__':
    unittest.main()
