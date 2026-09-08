import unittest
import copy
from pathlib import Path
import numpy as np
import trimesh
import yaml
from render.provisional import heart_surface
from render.acceptance import evaluate
from registration.calibration_gate import require_report
from render.window_camera import context,is_removed
from render.camera import sample_camera
from render.output import class_count_probability

ROOT=Path(__file__).resolve().parents[2]

class Round3Tests(unittest.TestCase):
    def test_bank_gate_rejects_changed_configuration(self):
        import json
        import tempfile
        configuration={'camera':{'seed':1,'resolution':[256,144]},'dissection':{},'atlas':{}}
        report={'acceptance':{'bank_generation_allowed':True,'batch_complete':True,'frames':128,'missing_required_bank_class_ids':[]},'render_configuration':configuration}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'comparison.json';path.write_text(json.dumps(report))
            changed=copy.deepcopy(configuration);changed['camera']['seed']=2
            require_report(path,changed)
            changed['camera']['resolution']=[512,288]
            with self.assertRaises(ValueError):require_report(path,changed)
            report['acceptance']['batch_complete']=False;path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):require_report(path)

    def test_count_prior_corrects_empirical_proposal(self):
        cfg=yaml.safe_load((ROOT/'configs/camera_prior.yaml').read_text())
        q=cfg['quality']['visible_class_count'];prior=q['soft_preference'];support=np.arange(q['min'],q['max']+1)
        p=np.array([prior['proposal_counts'][int(n)] for n in support],float)
        accept=np.array([class_count_probability(np.arange(1,n+1,dtype=np.uint8)[None,:],cfg) for n in support])
        expected=np.exp(-.5*((support-prior['mean'])/prior['sd'])**2)
        np.testing.assert_allclose(p*accept/np.sum(p*accept),expected/expected.sum())
        self.assertTrue(np.all((accept>=0)&(accept<=1)))

    def test_aperture_distance_scales_and_scope_remains_rigid(self):
        cfg=yaml.safe_load((ROOT/'configs/camera_prior.yaml').read_text());cfg['resolution']=[256,144]
        cfg['window_coupling']['center_target_probability']=1.;cfg['window_coupling']['jitter_sd_mm']=0.
        cfg['window_coupling']['short_side_occupancy']={'mean':.65,'sd':0.,'min':.5,'max':.8}
        landmarks={n:[0,0,0] for n in cfg['targets']['names']}
        ports=[{'position_mm':[100,0,0]}]
        window={'name':'test','anchor':'esophagus','port_index':0,'center_mm':[0,0,0],'effective_radius_mm':8.}
        ctx={'windows':[window],'exposed_landmarks':[],'removed':np.ones((21,21,21),bool),'volume':{'origin':np.array([-10,-10,-10]),'pitch':1.}}
        first=sample_camera(np.random.default_rng(12),ports,landmarks,cfg,ctx)
        window['effective_radius_mm']=16.
        second=sample_camera(np.random.default_rng(12),ports,landmarks,cfg,ctx)
        self.assertAlmostEqual(second['working_distance_mm'],first['working_distance_mm']*2)
        for p in [first,second]:
            np.testing.assert_allclose(np.array(p['tip_mm'])+p['working_distance_mm']*np.array(p['optical_axis_ras']),p['target_mm'],atol=1e-6)
            self.assertTrue(p['target_name'].startswith('window:'))

    def test_ipv_clearance_subtracted_after_dilation(self):
        heart=trimesh.creation.box(extents=[20,20,20]);vein=trimesh.creation.cylinder(radius=2,height=40,sections=24)
        obj=lambda m:{'v':m.vertices,'f':m.faces}
        cfg={'voxel_mm':1.,'margin_mm':2.,'closing_iterations':1,'ipv_clearance_mm':3.}
        v,f=heart_surface([obj(heart)],cfg,[obj(vein)])
        # Through-hole axis along Z; no proxy surface may cover its central clearance cylinder.
        self.assertGreater(len(f),0)
        self.assertGreater(np.min(np.linalg.norm(v[:,:2],axis=1)),3.)
        uncut,_=heart_surface([obj(heart)],cfg)
        self.assertLess(np.min(np.linalg.norm(uncut[:,:2],axis=1)),1.)

    def test_joint_acceptance_and_small_sample(self):
        ref=yaml.safe_load((ROOT/'assets/real_mask_statistics.yaml').read_text());cfg=yaml.safe_load((ROOT/'configs/acceptance.yaml').read_text())
        s={'frames':128,'presence_pct':[r['presence_pct'] for r in ref['classes']],'mean_area_pct':[r['mean_area_pct'] for r in ref['classes']],'visible_class_count':{'median':15},'background_summary':{'median_pct':18.}}
        self.assertTrue(evaluate(s,ref,cfg)['bank_generation_allowed'])
        s['frames']=127;self.assertFalse(evaluate(s,ref,cfg)['acceptance_passed']);s['frames']=128
        s['mean_area_pct'][10]=65.;self.assertFalse(evaluate(s,ref,cfg)['acceptance_passed'])
        with self.assertRaises(ValueError):require_report(None)

    def test_exposure_test_uses_actual_voxel_mask(self):
        v={'origin':np.array([0,0,0]),'pitch':1.};removed=np.zeros((5,5,5),bool);removed[2,2,2]=True
        ctx={'volume':v,'removed':removed}
        self.assertTrue(is_removed([2,2,2],ctx));self.assertFalse(is_removed([1,2,2],ctx));self.assertFalse(is_removed([-1,2,2],ctx))

if __name__=='__main__':unittest.main()
