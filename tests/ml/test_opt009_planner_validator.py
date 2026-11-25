"""Run: python -m unittest discover -s tests -p 'test_opt009_planner_validator.py'"""
import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'ml/scripts/generate_goemotions_large_scale.py'
spec = importlib.util.spec_from_file_location('generation_under_test', SCRIPT)
gen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gen
spec.loader.exec_module(gen)

class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.prompt = gen.load_spec()
        cls.jobs = gen.make_jobs(cls.cfg)

    def test_job_count_and_quotas(self):
        self.assertEqual(len(self.jobs), 4200)
        self.assertEqual(sum(len(j['slots']) for j in self.jobs), 42000)
        for label, quota in self.cfg['primary_label_quotas'].items():
            self.assertEqual(sum(len(j['slots']) for j in self.jobs if j['primary_label'] == label), quota)

    def test_planner_reproducibility(self):
        self.assertEqual(self.jobs, gen.make_jobs(self.cfg))
        self.assertEqual(gen.manifest_for(self.cfg, self.prompt, self.jobs), gen.manifest_for(self.cfg, self.prompt, gen.make_jobs(self.cfg)))

    def test_slot_contract(self):
        for j in self.jobs:
            self.assertEqual(len(j['slots']), self.cfg['examples_per_request'])
            self.assertEqual([x['scenario_slot'] for x in j['slots']], list(range(10)))
            self.assertEqual(len({(x['domain'], x['scenario']) for x in j['slots']}), 10)
            for x in j['slots']:
                self.assertIn(x['length'], gen.LENGTH_LIMITS)
                self.assertIn(x['difficulty'], {'easy','moderate','hard'})
                self.assertIn(x['voice'], self.cfg['voices'])
                self.assertTrue(x['semantic_contrast'] == 'none' or x['semantic_contrast'] in self.cfg['allowed_secondary_labels'])

    def make_batch(self, job):
        examples=[]
        for slot in job['slots']:
            low, high = gen.LENGTH_LIMITS[slot['length']]
            words = [f'unique{slot["scenario_slot"]}'] + ['ordinary'] * (low-1)
            examples.append(gen.Example(scenario_slot=slot['scenario_slot'], text=' '.join(words), secondary_labels=[]))
        return gen.Batch(examples=examples)

    def test_accept_valid_batch(self):
        job=self.jobs[0]
        gen.validate(self.make_batch(job), job, self.cfg)

    def test_reject_bad_lengths(self):
        job=self.jobs[0]
        b=self.make_batch(job)
        b.examples[0].text='only one'
        with self.assertRaisesRegex(ValueError, 'Length band violation'):
            gen.validate(b, job, self.cfg)

    def test_reject_bad_slot(self):
        job=self.jobs[0]
        b=self.make_batch(job)
        b.examples[0].scenario_slot=9
        with self.assertRaisesRegex(ValueError, 'Slot order'):
            gen.validate(b, job, self.cfg)

    def test_reject_primary_as_secondary(self):
        job=self.jobs[0]
        b=self.make_batch(job)
        b.examples[0].secondary_labels=[job['primary_label']]
        with self.assertRaisesRegex(ValueError, 'Unrecognized/primary'):
            gen.validate(b, job, self.cfg)

    def test_reject_duplicate(self):
        job=self.jobs[0]
        b=self.make_batch(job)
        # Can compare only equally-sized bands; force two slots to same valid text
        same_band=next((i,j) for i in range(10) for j in range(i+1,10) if job['slots'][i]['length']==job['slots'][j]['length'])
        i,j=same_band
        b.examples[j].text=b.examples[i].text.upper()
        with self.assertRaisesRegex(ValueError, 'In-batch duplicate'):
            gen.validate(b, job, self.cfg)

if __name__ == '__main__':
    unittest.main()
