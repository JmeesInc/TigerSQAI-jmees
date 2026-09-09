import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from render.calibration_report import aggregate
from render.generate import instruments
from registration.features import weighted_iou
from registration.calibration_gate import require_report

class Round4Tests(unittest.TestCase):
    def test_wall_is_an_inset_mesh_and_preserves_source(self):
        import trimesh
        from render.chest_wall import build_chest_wall
        mesh=trimesh.creation.box(extents=[100,100,100]);before=mesh.vertices.copy()
        wall=build_chest_wall([{'role':'rib','v':mesh.vertices}],[],{'enabled':True,'inset_mm':3,'mesh_edge_mm':10})
        self.assertEqual(wall['fine_id'],10)
        self.assertLess(np.max(np.abs(wall['v'])),50)
        self.assertTrue(trimesh.Trimesh(wall['v'],wall['f'],process=False).is_watertight)
        np.testing.assert_array_equal(mesh.vertices,before)

    def test_per_frame_anatomy_normalization_and_exclusions(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i,a in enumerate([[[0,1,2,3,10]],[[0,0,0,0,10]]]):
                p=Path(d)/f'{i}.png';Image.fromarray(np.array(a,np.uint8)).save(p);paths.append(p)
            s=aggregate(paths)
            self.assertEqual(s['mean_area_pct'][3],25)
            self.assertEqual(s['mean_area_pct'][10],75)
            self.assertEqual(s['mean_area_pct'][1],0)
            self.assertEqual(s['visible_class_count']['median'],1.5)
            self.assertEqual(s['background_summary']['median_pct'],50)
    def test_real_background_is_unknown_but_synthetic_holes_penalized(self):
        real=np.array([[0,0,3,3]],np.uint8);synth=np.array([[4,4,3,3]],np.uint8)
        self.assertEqual(weighted_iou(real,synth,[3,4],{3:1,4:1})['weighted_iou'],1)
        synth[0,3]=0
        self.assertEqual(weighted_iou(real,synth,[3,4],{3:1,4:1})['weighted_iou'],.5)
        self.assertLess(weighted_iou(real,synth,[3,4],{3:1,4:1},real_background_policy='compare')['weighted_iou'],.5)
    def test_disabled_instruments_do_not_consume_rng_or_require_geometry(self):
        self.assertEqual(instruments(None,None,None,{'enabled':False}),[])
    def test_round3_report_cannot_unlock_round4_bank(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json';p.write_text(json.dumps({'acceptance':{'bank_generation_allowed':True,'batch_complete':True,'frames':128,'missing_required_bank_class_ids':[]}}))
            with self.assertRaises(ValueError):require_report(p)

if __name__=='__main__':unittest.main()
