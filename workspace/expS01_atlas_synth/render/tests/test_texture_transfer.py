import unittest
import numpy as np
from render.texture_transfer import barycentric,chart_grid,sample_image

class TextureTransferTests(unittest.TestCase):
 def test_barycentric_surface_point(self):
  tri=np.array([[0.,0,0],[2.,0,0],[0,3.,0]])
  np.testing.assert_allclose(barycentric([.5,.75,0],tri),[.5,.25,.25])
 def test_uv_origin_and_bilinear(self):
  image=np.array([[0.,1.],[2.,3.]])
  np.testing.assert_allclose(sample_image(image,np.array([[0.,0.],[1,1],[.5,.5]])),[2,1,1.5])
 def test_chart_padding_and_corner_texels(self):
  weights,corners=chart_grid(8)
  self.assertTrue((weights>=0).all());np.testing.assert_allclose(weights.sum(1),1.)
  for i,(x,y) in enumerate(corners):np.testing.assert_allclose(weights[int(y)*8+int(x)],np.eye(3)[i])
if __name__=='__main__':unittest.main()
