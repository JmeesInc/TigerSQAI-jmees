import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from registration.features import describe,distances,weighted_iou,read_label
from registration.register import confidence,resized_camera
from registration.fit_priors import fit
from render.output import reject_reason
from render.provisional import add_proxies,heart_surface
import trimesh

ROOT=Path(__file__).resolve().parents[2]

class RegistrationTests(unittest.TestCase):
    def test_missing_capable_class_penalty_and_nuisance_exclusion(self):
        a=np.zeros((24,32),np.uint8);a[4:20,3:14]=3;a[4:20,17:28]=6
        b=a.copy();b[b==6]=0;c=a.copy();c[:3,:]=1
        ds=[describe(x) for x in [a,b,c]];bank={k:np.stack([d[k] for d in ds]) for k in ds[0]}
        w={'presence':1,'area':1,'centroid':1,'moment':1,'adjacency':1}
        d=distances(ds[0],bank,[3,6],w);self.assertEqual(d[0],0);self.assertGreater(d[1],d[0]);self.assertEqual(d[2],0)
        iou=weighted_iou(a,b,[3,6],{3:2,6:2});self.assertEqual(iou['weighted_iou'],.5)
        self.assertEqual(weighted_iou(a,c,[3,6],{3:2,6:2})['weighted_iou'],1)

    def test_moments_adjacency_and_missing_mask(self):
        a=np.full((10,20),3,np.uint8);a[:,10:]=6;d=describe(a)
        np.testing.assert_allclose(d['centroid'][3],[.25,.5]);self.assertTrue(d['adjacency'][3,6]);self.assertFalse(d['present'][8]);self.assertTrue(np.isfinite(d['moment']).all())

    def test_intrinsics_resize_pixel_centers(self):
        m={'resolution':[256,144],'camera':{'K':[[100,0,127.5],[0,120,71.5],[0,0,1]],'distortion':{'k1':-.05}}}
        k=np.array(resized_camera(m,[1024,576])['K']);np.testing.assert_allclose(k,[[400,0,511.5],[0,480,287.5],[0,0,1]])
        self.assertEqual(m['camera']['K'][0][0],100)

    def test_ambiguity_and_insufficient_evidence_rejected(self):
        cfg=yaml.safe_load((ROOT/'configs/registration.yaml').read_text());a={'weighted_iou':.85,'descriptor_distance':.05,'common_visible_classes':[3,4,5,6,10,11],'comparable_pixel_fraction':.8,'translation_from_best_mm':0,'angle_from_best_deg':0};b=a|{'weighted_iou':.84,'translation_from_best_mm':60}
        _,reasons,_=confidence([a,b],cfg);self.assertIn('ambiguous spatial hypotheses',reasons)
        _,reasons,_=confidence([a|{'common_visible_classes':[3]}],cfg);self.assertIn('too few comparable classes',reasons)

    def test_label_format_and_aspect_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.png';Image.new('RGB',(16,9)).save(p)
            with self.assertRaises(ValueError):read_label(p)
            Image.new('L',(16,9)).save(p)
            with self.assertRaises(ValueError):read_label(p,[10,10])

    def test_proxy_disable_and_heart_source_dilation(self):
        box=trimesh.creation.box(extents=[10,12,14]);o={'fine_id':0,'role':'proxy_source','name':'test','v':box.vertices,'f':box.faces}
        self.assertEqual(add_proxies([o],{'provisional':{'enabled':False}}),([],[]))
        v,f=heart_surface([o],{'voxel_mm':1,'margin_mm':2,'closing_iterations':1});self.assertGreater(len(f),0);self.assertGreater(np.ptp(v[:,0]),10)

    def test_foreground_class_count_filter(self):
        cfg=yaml.safe_load((ROOT/'configs/camera_prior.yaml').read_text());a=np.tile(np.arange(10,dtype=np.uint8),(24,24));reason=reject_reason(a,np.ones_like(a,bool),cfg,.8);self.assertEqual(reason,'visible class count')
        a=np.tile(np.arange(11,dtype=np.uint8),(24,24));self.assertIsNone(reject_reason(a,np.ones_like(a,bool),cfg,.8))

    def test_fit_gates_do_not_treat_duplicates_as_diversity(self):
        cfg=yaml.safe_load((ROOT/'configs/registration.yaml').read_text())
        with self.assertRaises(ValueError):fit([],{}, {},cfg)
        duplicates=[{'accepted':True,'top_k':[{'bank_id':'same'}]}]*30
        with self.assertRaises(ValueError):fit(duplicates,{}, {},cfg)

if __name__=='__main__':unittest.main()
