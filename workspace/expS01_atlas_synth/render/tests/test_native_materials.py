import unittest
import numpy as np
from render.native_materials import triangle_rows

class NativeMappingTests(unittest.TestCase):
 def test_roi_subset_preserves_distinct_slots(self):
  source=np.array([[0,1,2],[2,3,4],[5,6,7]])
  slots=np.array([4,2,8]);rows=triangle_rows(source,source[[2,0]])
  np.testing.assert_array_equal(slots[rows],[8,4])
 def test_modified_or_duplicate_topology_rejected(self):
  with self.assertRaises(ValueError):triangle_rows([[0,1,2]],[[0,2,1]])
  with self.assertRaises(ValueError):triangle_rows([[0,1,2],[0,1,2]],[[0,1,2]])
if __name__=='__main__':unittest.main()
