import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
import OpenEXR
from render.output import write_outputs,read_passes
from render.registration import apply_patient
from render.util import sha256


class OutputRegistrationTests(unittest.TestCase):
    def test_png_exr_roundtrip_without_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'label').mkdir();(path/'depth').mkdir()
            labels=np.arange(31,dtype=np.uint8)[None,:].repeat(4,axis=0)
            depth=np.arange(124,dtype=np.float32).reshape(4,31)*.123
            write_outputs(path,'test',labels,depth)
            image=Image.open(path/'label/test.png')
            self.assertEqual(image.mode,'L');np.testing.assert_array_equal(np.array(image),labels)
            with OpenEXR.File(str(path/'depth/test.exr'),separate_channels=True) as f:
                self.assertEqual(list(f.channels()),['Z'])
                np.testing.assert_array_equal(f.channels()['Z'].pixels,depth)

    def test_ct_mask_affine_and_tps_integration(self):
        # Analytic cuboid fixture tests coordinate handling, not synthetic anatomy.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'public_test_fixture.nii.gz'
            mask=np.zeros((10,12,14),np.uint8);mask[2:8,3:9,4:10]=1
            affine=np.array([[0,-2,0,50],[1,0,0,20],[0,0,3,-40],[0,0,0,1]],float)
            image=nib.Nifti1Image(mask,affine);image.header.set_xyzt_units('mm');nib.save(image,path)
            src=np.array([[0,0,0],[20,0,0],[0,20,0],[0,0,20],[20,20,20],[10,-10,5]],float)
            shift=np.array([2,4,6])
            patient={'landmark_pairs':[{'group':g,'atlas_ras_mm':s.tolist(),'ct_ras_mm':(s+shift).tolist()}
                       for s,g in zip(src,['skeleton','skeleton','trachea','trachea','aorta','aorta'])],
                     'masks':[{'name':'fixture','fine_id':3,'path':str(path),'sha256':sha256(path),
                               'license':'test fixture','source_url':'local analytic unit-test fixture'}]}
            objects=[{'fine_id':3,'name':'to_replace','v':src,'f':np.array([[0,1,2]])},
                     {'fine_id':14,'name':'to_warp','v':src,'f':np.array([[0,1,2]])}]
            result,ribs,landmarks,report=apply_patient(objects,{'3':src},{'carina':src[0]},patient)
            fine14=next(o for o in result if o['fine_id']==14)
            np.testing.assert_allclose(fine14['v'],src+shift,atol=1e-5)
            np.testing.assert_allclose(ribs['3'],src+shift,atol=1e-5)
            np.testing.assert_allclose(landmarks['carina'],shift,atol=1e-5)
            replacement=next(o for o in result if o['fine_id']==3)
            np.testing.assert_allclose(replacement['v'].min(axis=0),[33,21.5,-29.5])
            np.testing.assert_allclose(replacement['v'].max(axis=0),[45,27.5,-11.5])
            self.assertGreater(report['min_sampled_jacobian'],.99)


if __name__=='__main__':unittest.main()
