import unittest
from pathlib import Path
import numpy as np
import yaml
from render.camera import solve_scope,optical_axis,distortion_grid
from render.registration import ThinPlateSpline3D
from render.volume import cut_mask,build_volume,dissect

ROOT=Path(__file__).resolve().parents[2]


class GeometryTests(unittest.TestCase):
    def test_actual_shell_geometry_and_blood_option(self):
        import trimesh
        cfg=yaml.safe_load((ROOT/'configs/dissection.yaml').read_text())
        cfg['blood']['probability']=1.;cfg['resection']['probability']=1.
        cfg['phases']=[{'name':'closed','interval':[0,.15],'windows':[]},
                       {'name':'open','interval':[.15,1.], 'windows':[{'anchor':'test',
                        'offset_mm':[10,0,0],'radii_mm':[20,20,20]}]}]
        box=trimesh.creation.box(extents=[20,20,20])
        volume=build_volume([{'fine_id':6,'v':box.vertices,'f':box.faces}],
                            {'roi_mm':[[-35,-35,-35],[35,35,35]]},cfg)
        self.assertTrue(volume['fat'].any());self.assertTrue(volume['pleura'].any())
        self.assertFalse(np.any(volume['fat']&volume['pleura']))
        pre,pm=dissect(volume,{'test':[0,0,0]},cfg,0,np.random.default_rng(1))
        post,qm=dissect(volume,{'test':[0,0,0]},cfg,1,np.random.default_rng(1))
        self.assertEqual({o['fine_id'] for o in pre},{7,10,20})
        self.assertLess(qm['remaining_fat_voxels'],pm['remaining_fat_voxels'])
        self.assertIn(24,{o['fine_id'] for o in post})
        self.assertIn(25,{o['fine_id'] for o in post})

    def test_oblique_scope_cone_and_pivot(self):
        for theta in [0,30]:
            for azimuth in np.linspace(-3,3,13):
                p=np.array([100.,-40.,30.]);t=np.array([0.,0.,10.])
                tip,s,v,length,rotation=solve_scope(p,t,35,theta,azimuth)
                np.testing.assert_allclose(tip,p+length*s,atol=1e-8)
                np.testing.assert_allclose(t,tip+35*v,atol=1e-8)
                self.assertAlmostEqual(s@v,np.cos(np.deg2rad(theta)))
                np.testing.assert_allclose(v,optical_axis(s,theta,rotation),atol=1e-8)
        shaft=np.array([1.,0.,0.])
        cone=np.array([optical_axis(shaft,30,a) for a in np.linspace(0,2*np.pi,100)])
        np.testing.assert_allclose(cone[:,0],np.cos(np.pi/6))
        self.assertGreater(np.ptp(cone[:,1]),.99)
        self.assertGreater(np.ptp(cone[:,2]),.99)

    def test_identity_warp_and_integer_sampling(self):
        cfg=yaml.safe_load((ROOT/'configs/camera_prior.yaml').read_text())
        cfg['resolution']=[160,90];cfg['distortion']['enabled']=False
        camera={'K':[[100,0,79.5],[0,100,44.5],[0,0,1]]}
        x,y,valid=distortion_grid(np.random.default_rng(2),camera,cfg)
        yy,xx=np.mgrid[:90,:160]
        np.testing.assert_array_equal(xx,x);np.testing.assert_array_equal(yy,y)
        self.assertTrue(valid.all())

    def test_tps_affine_reproduction_and_nonlinearity(self):
        rng=np.random.default_rng(4);src=rng.normal(size=(12,3))*30
        target=src@np.diag([1.1,.9,1.2])+[10,20,-4]
        warp=ThinPlateSpline3D(src,target,0)
        query=rng.normal(size=(50,3))*20
        np.testing.assert_allclose(warp(query),query@np.diag([1.1,.9,1.2])+[10,20,-4],atol=1e-8)
        target[0]+=[3,-1,2]
        nonlinear=ThinPlateSpline3D(src,target,0)
        np.testing.assert_allclose(nonlinear(src),target,atol=1e-8)
        self.assertGreater(np.linalg.norm(nonlinear.weights),.01)

    def test_dissection_monotone_irregular_and_zero_before_incision(self):
        cfg=yaml.safe_load((ROOT/'configs/dissection.yaml').read_text())
        landmarks=yaml.safe_load((ROOT/'configs/atlas.yaml').read_text())['landmarks_mm']
        shape=(41,41,81)
        vol={'fat':np.ones(shape,bool),'noise':np.random.default_rng(9).normal(size=shape),
             'origin':np.array([-100,-100,-180]),'pitch':5.}
        previous=np.zeros(shape,bool)
        for t in np.linspace(0,1,31):
            removed,_=cut_mask(vol,landmarks,cfg,float(t))
            self.assertFalse(np.any(previous&~removed))
            if t<=.15:self.assertFalse(removed.any())
            previous=removed
        no_noise=vol|{'noise':np.zeros(shape)}
        smooth,_=cut_mask(no_noise,landmarks,cfg,.6)
        rough,_=cut_mask(vol,landmarks,cfg,.6)
        self.assertTrue(np.any(smooth!=rough))


if __name__=='__main__':unittest.main()
