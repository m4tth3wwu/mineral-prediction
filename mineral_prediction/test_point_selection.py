import unittest
import numpy as np
from select_mineral_points import clusters,buffered_roles

class SpatialSelectionTests(unittest.TestCase):
    def test_connected_chain_stays_in_one_group(self):
        # Endpoints exceed the radius, but a connecting occurrence joins them.
        labels=clusters(np.array([[0.,0.],[9.,0.],[18.,0.],[50.,0.]]),10)
        self.assertEqual(len(set(labels[:3])),1)
        self.assertNotEqual(labels[0],labels[3])

    def test_guard_boundary_and_held_group(self):
        xy=np.array([[0.,0.],[5.,0.],[25.,0.],[26.,0.]])
        labels=np.array([0,0,1,2])
        self.assertEqual(buffered_roles(xy,labels,0,20).tolist(),['test','test','guard_excluded','train'])

if __name__=='__main__':unittest.main()
