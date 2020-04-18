from numpy import ndarray, array, asarray, dot, cross, cov, array, finfo, min as npmin, max as npmax
from numpy.linalg import eigh, norm
from scipy.spatial import ConvexHull
import sys


########################################################################################################################
# from https://github.com/pboechat/pyobb
# adapted from: http://jamesgregson.blogspot.com/2011/03/latex-test.html
########################################################################################################################
class OBB:
    def __init__(self):
        self.rotation = None
        self.min = None
        self.max = None

    def transform(self, point):
        return dot(array(point), self.rotation)

    @property
    def get_min(self):
      return self.transform(self.min)

    @property
    def get_max(self):
      return self.transform(self.max)

    @property
    def centroid(self):
        return self.transform((self.min + self.max) / 2.0)

    @property
    def extents(self):
        # ARD: bug fix. Min/Max are already transformed
        return abs(self.transform((self.max - self.min) / 2.0))

    @property
    def points(self):
        return [
            # upper cap: ccw order in a right-hand system
            # 0: rightmost, topmost, farthest
            self.transform((self.max[0], self.max[1], self.min[2])),
            # 1: leftmost, topmost, farthest
            self.transform((self.min[0], self.max[1], self.min[2])),
            # 2: leftmost, topmost, closest
            self.transform((self.min[0], self.max[1], self.max[2])),
            # 3: rightmost, topmost, closest
            self.transform(self.max),
            # lower cap: cw order in a right-hand system
            # 4: leftmost, bottommost, farthest
            self.transform(self.min),
            # 5: rightmost, bottommost, farthest
            self.transform((self.max[0], self.min[1], self.min[2])),
            # 6: rightmost, bottommost, closest
            self.transform((self.max[0], self.min[1], self.max[2])),
            # 7: leftmost, bottommost, closest
            self.transform((self.min[0], self.min[1], self.max[2])),
        ]

    @classmethod
    def build_from_covariance_matrix(cls, covariance_matrix, points):
        if not isinstance(points, ndarray):
            points = array(points, dtype=float)
        assert points.shape[1] == 3

        obb = OBB()

        _, eigen_vectors = eigh(covariance_matrix)

        def try_to_normalize(v):
            n = norm(v)
            if n < finfo(float).resolution:
                raise ZeroDivisionError
            return v / n

        r = try_to_normalize(eigen_vectors[:, 0])
        u = try_to_normalize(eigen_vectors[:, 1])
        f = try_to_normalize(eigen_vectors[:, 2])

        obb.rotation = array((r, u, f)).T

        # apply the rotation to all the position vectors of the array
        # TODO : this operation could be vectorized with tensordot
        p_primes = asarray([obb.rotation.dot(p) for p in points])
        obb.min = npmin(p_primes, axis=0)
        # print("obb p_primes",p_primes)
        obb.max = npmax(p_primes, axis=0)

        return obb

    @classmethod
    def build_from_triangles(cls, points, triangles):
        for point in points:
            if len(point) != 3:
                raise Exception('points have to have 3-elements')

        weighed_mean = array([0, 0, 0], dtype=float)
        area_sum = 0
        c00 = c01 = c02 = c11 = c12 = c22 = 0
        for i in range(0, len(triangles), 3):
            p = array(points[triangles[i]], dtype=float)
            q = array(points[triangles[i + 1]], dtype=float)
            r = array(points[triangles[i + 2]], dtype=float)
            mean = (p + q + r) / 3.0
            area = norm(cross((q - p), (r - p))) / 2.0
            weighed_mean += mean * area
            area_sum += area
            c00 += (9.0 * mean[0] * mean[0] + p[0] * p[0] + q[0] * q[0] + r[0] * r[0]) * (area / 12.0)
            c01 += (9.0 * mean[0] * mean[1] + p[0] * p[1] + q[0] * q[1] + r[0] * r[1]) * (area / 12.0)
            c02 += (9.0 * mean[0] * mean[2] + p[0] * p[2] + q[0] * q[2] + r[0] * r[2]) * (area / 12.0)
            c11 += (9.0 * mean[1] * mean[1] + p[1] * p[1] + q[1] * q[1] + r[1] * r[1]) * (area / 12.0)
            c12 += (9.0 * mean[1] * mean[2] + p[1] * p[2] + q[1] * q[2] + r[1] * r[2]) * (area / 12.0)

        weighed_mean /= area_sum
        c00 /= area_sum
        c01 /= area_sum
        c02 /= area_sum
        c11 /= area_sum
        c12 /= area_sum
        c22 /= area_sum

        c00 -= weighed_mean[0] * weighed_mean[0]
        c01 -= weighed_mean[0] * weighed_mean[1]
        c02 -= weighed_mean[0] * weighed_mean[2]
        c11 -= weighed_mean[1] * weighed_mean[1]
        c12 -= weighed_mean[1] * weighed_mean[2]
        c22 -= weighed_mean[2] * weighed_mean[2]

        covariance_matrix = ndarray(shape=(3, 3), dtype=float)
        covariance_matrix[0, 0] = c00
        covariance_matrix[0, 1] = c01
        covariance_matrix[0, 2] = c02
        covariance_matrix[1, 0] = c01
        covariance_matrix[1, 1] = c11
        covariance_matrix[1, 2] = c12
        covariance_matrix[2, 0] = c02
        covariance_matrix[1, 2] = c12
        covariance_matrix[2, 2] = c22
        print("cov matrix",covariance_matrix)

        return OBB.build_from_covariance_matrix(covariance_matrix, points)

    @classmethod
    def build_from_points(cls, points):
        if not isinstance(points, ndarray):
            points = array(points, dtype=float)
        assert points.shape[1] == 3, 'points have to have 3-elements'
        # no need to store the covariance matrix
        return OBB.build_from_covariance_matrix(cov(points, y=None, rowvar=0, bias=1), points)


    ####################
    # New obb functions
    ####################
    @classmethod
    def distance_3d(cls, pt1, pt2):
       return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

    # ARD: Not working 
    @classmethod
    def obbvolume(cls, bpnts):
      d = []
      # topmost, farthest
      d.append(OBB.distance_3d(bpnts[0], bpnts[1]))
      # topmost, closest
      d.append(OBB.distance_3d(bpnts[2], bpnts[3]))
      # topmost, leftmost
      d.append(OBB.distance_3d(bpnts[1], bpnts[2]))
      # topmost, rightmost
      d.append(OBB.distance_3d(bpnts[0], bpnts[3]))
      assert  abs(d[0]-d[1])/d[1] < 0.05 , 'not same distances'
      assert  abs(d[2]-d[3])/d[3] < 0.05 , 'not same distances'
      # bottommost, farthest
      d.append(OBB.distance_3d(bpnts[0], bpnts[1]))
      # bottommost, closest
      d.append(OBB.distance_3d(bpnts[2], bpnts[3]))
      # bottommost, leftmost
      d.append(OBB.distance_3d(bpnts[1], bpnts[2]))
      # bottommost, rightmost
      d.append(OBB.distance_3d(bpnts[0], bpnts[3]))
      assert  abs(d[4]-d[5])/d[5] < 0.05 , 'not same distances'
      assert  abs(d[6]-d[7])/d[7] < 0.05 , 'not same distances'
      # side, rightmost, farthest
      d.append(OBB.distance_3d(bpnts[0], bpnts[5]))
      # side, rightmost, closest
      d.append(OBB.distance_3d(bpnts[3], bpnts[6]))
      # side, leftmost, farthest
      d.append(OBB.distance_3d(bpnts[1], bpnts[5]))
      # side, leftmost, clostest
      d.append(OBB.distance_3d(bpnts[2], bpnts[7]))
      assert  abs(d[4]-d[5])/d[5] < 0.05 , 'not same distances'
      assert  abs(d[6]-d[7])/d[7] < 0.05 , 'not same distances'
      print("distances: ", d)
  
      vol = d[0] * d[4] * d[8]
      return vol
  
    @classmethod
    def obb_volume(cls, bb):
      # bb = self.cluster['obb'] 
      if bb is None:
        print("no bounding box ")
        return None
      try:
        hull = ConvexHull(bb.points)
      except:
        return None
      return hull.volume

    @classmethod
    def obb_overlap(cls, bb, bb2):
      from scipy.spatial import ConvexHull
      # bb = self.cluster['obb'] 
      # bb2 = pc_cluster.cluster['obb']
      if bb == None or bb2 == None:
        print("no bounding box ")
        return None, None
      try:
        # print("obb_vol",OBB.obb_volume(bb))
        # print("obb_vol",OBB.obb_volume(bb2))
        v1 = OBB.obb_volume(bb)
        # v1b = OBB.obbvolume(bb)
        v2 = OBB.obb_volume(bb2)
        # v2b = OBB.obbvolume(bb2)
        # combined_pnts = bb.points+bb2.points
        print("combined_pnts: ",combined_pnts)
        hull = ConvexHull(combined_pnts)
        # print("volume: ",hull.volume)
        v3 = hull.volume
      except:
        # See above for failed examples of better QhullError handling...
        return None, None
      # print("v1,2: ",v1, v2)
      if v1 == None or v2 == None:
        return None, None
      # take min in case v1 >>> v2 or v2 >>> v1
      max_overlp = max(v1,v2) / hull.volume
      min_overlp = min(v1,v2) / hull.volume
      # print("obb vols:", v1,v1b, v2, v2b, v3, overlp, len(hull.vertices))
      # print("obb vols:", v1,v2, v3, overlp, len(hull.vertices))
      # print("max,min", max_overlp, min_overlp)
      return max_overlp, min_overlp

    @classmethod
    def in_obb(cls, bb, pt):
      def pyramid_vol(bb,v0,v1,v2,v3,pt):
        hull = ConvexHull([bb.points[v0], bb.points[v1], bb.points[v2], bb.points[v3], pt])
        return hull.volume

#      # bug: Max, Min need to be transformed first.
#      # do simple quick check first
#      min = bb.min
#      max = bb.max
#      if (pt[0] <= min[0] or pt[1] <= min[1] or pt[2] <= min[2]
#       or pt[0] >= max[0] or pt[1] >= max[1] or pt[2] >= max[2]):
#        print("Point outside OBB")
#        print("min", min)
#        print("max", max)
#        print("pt ", pt)
#        return False

      obb_vol = OBB.obb_volume(bb)
      obb_pts = bb.points

      try:
        # volume of each side + point
        # 2: leftmost, topmost, closest
        # topmost:    vertices 0123
        vol_top = pyramid_vol(bb,0,1,2,3,pt)
        # bottommost: vertices 4567
        vol_bot = pyramid_vol(bb,4,5,6,7,pt)
        # leftmost:   vertices 1247
        vol_left = pyramid_vol(bb,1,2,4,7,pt)
        # rightmost:  vertices 0356
        vol_right = pyramid_vol(bb,0,3,5,6,pt)
        # farthest:   vertices 0145
        vol_far = pyramid_vol(bb,0,1,4,5,pt)
        # closest:    vertices 2367
        vol_close = pyramid_vol(bb,2,3,6,7,pt)
      # except scipy.spatial.qhull.QhullError as e:
      except:
        # on boundary(?)
        errstr = sys.exc_info()[1]
        print(errstr[:100])
        return True, None
        # if (sys.exc_info()[0] != "<class 'scipy.spatial.qhull.QhullError'>"):
        # if (sys.exc_info()[0] != 'scipy.spatial.qhull.QhullError'):
        # if sys.exc_info()[1][:len('QH6154')] != 'QH6154':
        # errstr = sys.exc_info()[1]
        # if errstr[12:len('QH6154')+12] != 'QH6154':
          # print("QhullError error:", sys.exc_info()[0])
          # print("len errstr: ",len(errstr))
          # print("errstr1: ",errstr[1:len('QH6154')+1])
          # print("errstr2: ",errstr[2:len('QH6154')+2])
          # print("errstr3: ",errstr)
          # print(sys.exc_info()[1][:len('QH6154')])
          # print("Full QhullError error:", sys.exc_info())
      tot_vol = vol_top + vol_bot + vol_left + vol_right + vol_far + vol_close
      # APPROX0 = 0.0001
      APPROX0 = 0.000000001
      # if abs((tot_vol - obb_vol)/obb_vol) > APPROX0:
      # ARD BUG: multiple clusters are assigned to same grip????
      if (tot_vol / obb_vol) > 1.1:
        # if point is inside OBB, the volumes should be equal
        # print("vol dif:", abs(tot_vol - obb_vol), tot_vol, obb_vol, (tot_vol / obb_vol))
        return False, (tot_vol / obb_vol) 
      else:
        # if (vol_top <= APPROX0 or vol_bot <= APPROX0 or vol_left <= APPROX0 or vol_right <= APPROX0 or vol_far <= APPROX0 or vol_close <= APPROX0):
          # print("Point on side of OBB")
        # print("vols:", vol_top, vol_bot, vol_left, vol_right, vol_far, vol_close)
        # print("vol difs:", abs(tot_vol - obb_vol), tot_vol, obb_vol)
        return True, (tot_vol / obb_vol)


#      pt1 = bb.points
#      # pnts = array(pt, dtype=float) + pt1
#      pnts = pt1
#      # print("pnts:", pnts)
#      bb2 = OBB.build_from_points(pnts)
#      pt2 = bb2.points
#      print("pt1 =", pt1)
#      print("pt2 =", pt2)
#      count = 0
#      for i in range(8):
#        if pt1[i][0] == pt2[i][0] and pt1[i][1] == pt2[i][1] and pt1[i][2] == pt2[i][2]:
#          count += 1
#      if count == 8:
#        print("in bounding box")
#        return True
#      print("Not in bounding box")
#      return False


