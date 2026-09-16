import unittest
from contextlib import contextmanager
from uuid import uuid4
from pathlib import Path
from unittest.mock import patch
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box
import PPP_binary_lithology_v3 as v3


@contextmanager
def cache_fixture_directory():
    # Python's mode-0700 Windows temporary directory ACL is unusable under
    # this desktop sandbox. A normal workspace directory inherits its ACL.
    path=v3.ROOT/'output'/('cache_test_'+uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        for child in path.iterdir():child.unlink()
        path.rmdir()


class GeometryTests(unittest.TestCase):
    def test_internal_seam_and_water_edge_are_excluded(self):
        lith=gpd.GeoDataFrame({'Symbol':[v3.PINK,v3.PINK,'Sedimentary, clastic','Water']},
            geometry=[box(0,0,1,1),box(1,0,2,1),box(0,1,2,2),box(0,-1,2,0)],crs='ESRI:102039')
        pink,other,shared,old,stats=v3.binary_geometry(lith)
        self.assertAlmostEqual(pink.area,2)
        self.assertAlmostEqual(shared.length,2)
        self.assertTrue(shared.equals(shapely.LineString([(0,1),(2,1)])))
        self.assertGreater(old.length,shared.length)
        distances=v3.nearest_lines(shapely.points([1,1],[.5,0]),shared)*1000
        np.testing.assert_allclose(distances,[.5,1])

    def test_overlap_priority_and_unknown_not_contact(self):
        lith=gpd.GeoDataFrame({'Symbol':[v3.PINK,'Sedimentary, clastic','Unknown']},
            geometry=[box(0,0,2,2),box(1,0,3,2),box(0,2,2,3)],crs='ESRI:102039')
        pink,other,shared,_,stats=v3.binary_geometry(lith)
        self.assertAlmostEqual(pink.intersection(other).area,0)
        self.assertAlmostEqual(shared.length,2)
        self.assertAlmostEqual(stats['class_overlap_before_priority_km2'],2e-6)


class RegionTests(unittest.TestCase):
    def test_region_assignment_matches_nearest_center(self):
        events=pd.DataFrame({'metric_x':[0,1,100,101,0,1,100,101],
                             'metric_y':[0,1,0,1,100,101,100,101]})
        centers,regions=v3.compact_regions(events,events)
        xy=events[['metric_x','metric_y']].to_numpy()
        labels=((xy[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
        for point,label in zip(shapely.points(xy),labels):
            self.assertTrue(regions[label].covers(point))
        for region in regions:
            self.assertTrue(region.is_valid)
            self.assertEqual(region.geom_type,'Polygon')


class CacheTests(unittest.TestCase):
    def test_verified_reuse_and_corruption_invalidation(self):
        with cache_fixture_directory() as directory:
            cache=Path(directory)
            names=['grid.csv','events.csv','external.csv']
            for name in names:pd.DataFrame({'x':[1]}).to_csv(cache/name,index=False)
            signature={'version':'test','raw':'hash'}
            v3.save_json(cache/'manifest.json',{'signature':signature,
                         'outputs':{n:v3.digest(cache/n) for n in names}})
            with patch.object(v3,'feature_signature',return_value=signature):
                with patch.object(v3.ppp,'load_features',side_effect=RuntimeError('rebuild requested')) as build:
                    frames=v3.load_features(cache,cache,None)
                    self.assertEqual(len(frames),3)
                    build.assert_not_called()
                    (cache/'grid.csv').write_text('x\n2\n',encoding='utf-8')
                    with self.assertRaisesRegex(RuntimeError,'rebuild requested'):
                        v3.load_features(cache,cache,None)

    def test_changed_rule_invalidates_cache(self):
        with cache_fixture_directory() as directory:
            cache=Path(directory)
            v3.save_json(cache/'manifest.json',{'signature':{'version':'old'},'outputs':{}})
            with patch.object(v3,'feature_signature',return_value={'version':'new'}):
                with patch.object(v3.ppp,'load_features',side_effect=RuntimeError('rebuild requested')):
                    with self.assertRaisesRegex(RuntimeError,'rebuild requested'):
                        v3.load_features(cache,cache,None)

if __name__=='__main__':unittest.main()
