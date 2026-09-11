import unittest
from pathlib import Path
import numpy as np
import yaml
from render.alignment import boundary_metrics
from render.replay_rgb import saved_grid
from render.camera import distortion_grid

ROOT=Path(__file__).resolve().parents[2]
class Tier1Tests(unittest.TestCase):
    def test_boundary_fixture_detects_shift(self):
        a=np.zeros((40,80),np.uint8);a[:,40:]=3
        rgb=np.repeat((a>0)[:,:,None]*np.uint8(220),3,axis=2)
        self.assertTrue(boundary_metrics(a,rgb)['edge_evidence_pass'])
        self.assertFalse(boundary_metrics(a,np.roll(rgb,4,axis=1))['edge_evidence_pass'])
    def test_replay_grid_is_identical(self):
        cfg=yaml.safe_load((ROOT/'configs/camera_prior.yaml').read_text());cfg['resolution']=[256,144]
        pose={'K':[[160,0,127.5],[0,160,71.5],[0,0,1]]}
        first=distortion_grid(np.random.default_rng(23),pose,cfg)
        second=saved_grid(pose,[256,144],cfg['distortion']['inverse_iterations'])
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    def test_material_rankings_and_missing_measurements_explicit(self):
        c=yaml.safe_load((ROOT/'configs/materials.yaml').read_text())['classes']
        self.assertGreater(c[24]['bump_strength'],c[10]['bump_strength'])
        self.assertGreater(c[9]['noise_scale'],c[10]['noise_scale'])
        self.assertGreater(c[11]['roughness'],c[28]['roughness'])
        self.assertLess(c[12]['specular'],c[23]['specular'])
        self.assertEqual(set(c),set(range(31)))

if __name__=='__main__':unittest.main()
