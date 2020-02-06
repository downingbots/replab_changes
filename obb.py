from numpy import ndarray, array, asarray, dot, cross, cov, array, finfo, min as npmin, max as npmax
from numpy.linalg import eigh, norm


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
    def centroid(self):
        return self.transform((self.min + self.max) / 2.0)

    @property
    def extents(self):
        return abs(self.transform((self.max - self.min) / 2.0))

    @property
    def points3d(self):
        return [
            # upper cap: ccw order in a right-hand system
            # rightmost, topmost, farthest
            [self.max[0], self.max[1], self.min[2]],
            # leftmost, topmost, farthest
            [self.min[0], self.max[1], self.min[2]],
            # leftmost, topmost, closest
            [self.min[0], self.max[1], self.max[2]],
            # rightmost, topmost, closest
            self.max,
            # lower cap: cw order in a right-hand system
            # leftmost, bottommost, farthest
            self.min,
            # rightmost, bottommost, farthest
            [self.max[0], self.min[1], self.min[2]],
            # bottommost, closest
            [self.max[0], self.min[1], self.max[2]],
            # leftmost, bottommost, closest
            [self.min[0], self.min[1], self.max[2]],
        ]
 
    @property
    def points(self):
        return [
            # upper cap: ccw order in a right-hand system
            # rightmost, topmost, farthest
            self.transform((self.max[0], self.max[1], self.min[2])),
            # leftmost, topmost, farthest
            self.transform((self.min[0], self.max[1], self.min[2])),
            # leftmost, topmost, closest
            self.transform((self.min[0], self.max[1], self.max[2])),
            # rightmost, topmost, closest
            self.transform(self.max),
            # lower cap: cw order in a right-hand system
            # leftmost, bottommost, farthest
            self.transform(self.min),
            # rightmost, bottommost, farthest
            self.transform((self.max[0], self.min[1], self.min[2])),
            # rightmost, bottommost, closest
            self.transform((self.max[0], self.min[1], self.max[2])),
            # leftmost, bottommost, closest
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
