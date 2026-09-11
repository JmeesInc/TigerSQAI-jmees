import copy,unittest
from pathlib import Path
import numpy as np
import yaml
from render.chroma import material_linear_color
from render.configure_chroma import apply_chroma
from render.tissue_patterns import vascular_tile,texture_axes,texture_transverse
ROOT=Path(__file__).resolve().parents[2]

class TissueMaterialTests(unittest.TestCase):
    def test_measurements_are_loaded_not_placeholder_hues(self):
        cfg=yaml.safe_load((ROOT/'configs/materials.yaml').read_text());rows=yaml.safe_load((ROOT/'assets/real_class_chroma.yaml').read_text())['classes']
        for r in rows:
            p=cfg['classes'][r['fine_id']];c=np.array([r['chroma_r'],r['chroma_g'],r['chroma_b']]);np.testing.assert_allclose(p['base_chroma_rgb'],c/c.sum())
        self.assertEqual(cfg['classes'][18]['vascular_strength'],0)
        self.assertEqual(cfg['classes'][1]['vascular_strength'],0)
    def test_color_conversion_roundtrips_chroma(self):
        p={'base_chroma_rgb':[.462,.255,.283],'base_value_srgb':.65}
        linear=np.array(material_linear_color(p));srgb=np.where(linear<=.0031308,linear*12.92,1.055*linear**(1/2.4)-.055)
        np.testing.assert_allclose(srgb/srgb.sum(),p['base_chroma_rgb'])
    def test_vascular_texture_is_deterministic_multiscale_not_constant(self):
        a=vascular_tile(128,611);b=vascular_tile(128,611);np.testing.assert_array_equal(a,b)
        self.assertGreater(a.max(),.5);self.assertLess((a>.1).mean(),.6);self.assertGreater((a>.1).mean(),.01)
        self.assertFalse(np.array_equal(a,vascular_tile(128,612)))
    def test_principal_coordinates_do_not_change_vertices(self):
        rng=np.random.default_rng(33);v=rng.normal(size=(100,3))*[2,3,40];before=v.copy()
        axial=texture_axes(v);transverse=texture_transverse(v)
        np.testing.assert_array_equal(v,before);self.assertGreater(axial.std(),10*transverse.std())

if __name__=='__main__':unittest.main()
