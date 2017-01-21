////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public interface IParametricUV
    {
        /// <summary>
        /// Get a 3D point on this parametric surface.
        /// Parametric surfaces are only meaningfully defined in the range [0,1] in both U and V.
        /// </summary>
        /// <param name="u">The U location on the surface.</param>
        /// <param name="v">The V location on the surface.</param>
        /// <returns>A 3D point in object local space.</returns>
        Point3D GetPointUV(double u, double v);
    }
    public interface IParametricUVW
    {
        /// <summary>
        /// Get a 3D point within a parametric volume.
        /// Parametric volumes are only meaningfully defined in the range [0,1] in U, V and W.
        /// </summary>
        /// <param name="u">The U location in the volume.</param>
        /// <param name="v">The V location in the volume.</param>
        /// <param name="w">The W location in the volume.</param>
        /// <returns>A 3D point in object local space.</returns>
        Point3D GetPointUVW(double u, double v, double w);
    }
    interface IRayTest
    {
        /// <summary>
        /// Object specific ray intersection test.
        /// </summary>
        /// <param name="origin">Origin of object space ray.</param>
        /// <param name="direction">Direction of object space ray.</param>
        /// <returns>The positive distance along the ray direction to the first intersection (or +inf for no intersection was found).</returns>
        double RayTestDistance(Point3D origin, Vector3D direction);
        /// <summary>
        /// Object specific intersection normal.
        /// We assume this ray intersects and has been tested previously.
        /// </summary>
        /// <param name="origin">Origin of object space ray.</param>
        /// <param name="direction">Direction of object space ray.</param>
        /// <returns>The normal of the intersection point (NOT guaranteed to be a unit vector).</returns>
        Vector3D RayTestNormal(Point3D origin, Vector3D direction);
    }
    public class BezierPatch : IParametricUV
    {
        public BezierPatch()
        {
            // Define the hull for the patch.
            const double h = 0.5;
            hull = new Point3D[16]
            {
                new Point3D(-1, 0, -1), new Point3D(-h, 0, -1), new Point3D(+h, 0, -1), new Point3D(+1, 0, -1),
                new Point3D(-1, 0, -h), new Point3D(-h, 4, -h), new Point3D(+h, -4, -h), new Point3D(+1, 0, -h),
                new Point3D(-1, 0, +h), new Point3D(-h, -4, +h), new Point3D(+h, 4, +h), new Point3D(+1, 0, +h),
                new Point3D(-1, 0, +1), new Point3D(-h, 0, +1), new Point3D(+h, 0, +1), new Point3D(+1, 0, +1),
            };
        }
        public BezierPatch(Point3D[] hull)
        {
            this.hull = hull;
        }
        public Point3D GetPointUV(double u, double v)
        {
            // The Bernstein polynomial factors.
            double nu = 1 - u;
            double[] bu = new double[4] { nu * nu * nu, 3 * u * nu * nu, 3 * u * u * nu, u * u * u };
            double nv = 1 - v;
            double[] bv = new double[4] { nv * nv * nv, 3 * v * nv * nv, 3 * v * v * nv, v * v * v };
            // Compute the UV point.
            Point3D acc = new Point3D(0, 0, 0);
            for (int j = 0; j < 4; ++j)
            {
                for (int i = 0; i < 4; ++i)
                {
                    acc = MathHelp.Add(acc, MathHelp.Multiply(hull[i + j * 4], bu[i] * bv[j]));
                }
            }
            return acc;
        }
        Point3D[] hull = null;
    }
    public class Cube : IParametricUVW, IRayTest
    {
        public Point3D GetPointUVW(double u, double v, double w)
        {
            return new Point3D(-1 + u * 2, -1 + v * 2, -1 + w * 2);
        }
        public double RayTestDistance(Point3D origin, Vector3D direction)
        {
            double best_lambda = Double.PositiveInfinity;
            Vector3D best_normal;
            for (int face_index = 0; face_index < 6; ++face_index)
            {
                double lambda = IntersectPlane(origin, direction, face_normal[face_index], 1);
                if (lambda < 0 || lambda > best_lambda) continue;
                Point3D point = origin + lambda * direction;
                // Check that the point is inside every other plane.
                bool use_face = true;
                for (int check_face = 0; check_face < 6; ++check_face)
                {
                    if (face_index == check_face) continue;
                    double inside = MathHelp.Dot(point, face_normal[check_face]) - 1;
                    if (inside > 0)
                    {
                        use_face = false;
                        break;
                    }
                }
                if (use_face)
                {
                    best_lambda = lambda;
                    best_normal = face_normal[face_index];
                }
            }
            return best_lambda;
        }
        public Vector3D RayTestNormal(Point3D origin, Vector3D direction)
        {
            double best_lambda = Double.PositiveInfinity;
            Vector3D best_normal = new Vector3D(0, 0, 0);
            for (int face_index = 0; face_index < 6; ++face_index)
            {
                double lambda = IntersectPlane(origin, direction, face_normal[face_index], 1);
                if (lambda < 0 || lambda > best_lambda) continue;
                Point3D point = origin + lambda * direction;
                // Check that the point is inside every other plane.
                bool use_face = true;
                for (int check_face = 0; check_face < 6; ++check_face)
                {
                    if (face_index == check_face) continue;
                    double inside = MathHelp.Dot(point, face_normal[check_face]) - 1;
                    if (inside > 0)
                    {
                        use_face = false;
                        break;
                    }
                }
                if (use_face)
                {
                    best_lambda = lambda;
                    best_normal = face_normal[face_index];
                }
            }
            return best_normal;
        }
        double IntersectPlane(Point3D origin, Vector3D direction, Vector3D plane_normal, double plane_distance)
        {
            return (plane_distance - MathHelp.Dot(plane_normal, origin)) / MathHelp.Dot(plane_normal, direction);
        }
        static Vector3D[] face_normal = {
            new Vector3D(-1,0,0), new Vector3D(+1,0,0),
            new Vector3D(0,-1,0), new Vector3D(0,+1,0),
            new Vector3D(0,0,-1), new Vector3D(0,0,+1),
        };
    }
    public class Cylinder : IParametricUV
    {
        public Point3D GetPointUV(double u, double v)
        {
            // The central axis of the sphere points through world Y.
            // The U direction defines latitude and sweeps a full circle for 0 <= u <= 1.
            // The V direction defines linear distance along Y.
            double ucos = Math.Cos(u * Math.PI * 2);
            double usin = Math.Sin(u * Math.PI * 2);
            return new Point3D(-usin, -1 + v * 2, ucos);
        }
    }
    /// <summary>
    /// Plane in XZ.
    /// Note that for the purposes of parametric definitions this plane is bounded [-1,+1] in X and Z.
    /// The raytracer definition of this plane is infinite in the XZ plane.
    /// </summary>
    public class Plane : IParametricUV, IRayTest
    {
        public Point3D GetPointUV(double u, double v)
        {
            return new Point3D(-1 + u * 2, 0, -1 + v * 2);
        }
        /// <summary>
        /// For plane intersections we have the following possibilities:
        /// - Ray not parallel to plane; results in single intersection point.
        /// - Ray parallel to plane in half-space above (+ve); results in divide by zero and +Inf.
        /// - Ray parallel to plane in half-space behind (-ve); results in divide by zero and -Inf.
        /// </summary>
        /// <param name="origin">Origin of object space ray.</param>
        /// <param name="direction">Direction of object space ray.</param>
        /// <returns>The distance to the intersection point as described.</returns>
        public double RayTestDistance(Point3D origin, Vector3D direction)
        {
            return (PLANE_DISTANCE - MathHelp.Dot(PLANE_NORMAL, origin)) / MathHelp.Dot(PLANE_NORMAL, direction);
        }
        /// <summary>
        /// The normal of the plane is constant for all points in space.
        /// </summary>
        /// <param name="origin">Origin of object space ray.</param>
        /// <param name="direction">Direction of object space ray.</param>
        /// <returns>The normal of the plane.</returns>
        public Vector3D RayTestNormal(Point3D origin, Vector3D direction)
        {
            // Assume the ray hits, all normals are the plane normal.
            return PLANE_NORMAL;
        }
        const double PLANE_DISTANCE = 0;
        static Vector3D PLANE_NORMAL = new Vector3D(0, 1, 0);
    }
    /// <summary>
    /// Sphere of unit radius.
    /// The parametric definition of this sphere is oriented with the poles in Y.
    /// The "seam" of the sphere is deliberately behind the sphere in +Z.
    /// </summary>
    public class Sphere : IParametricUV, IRayTest
    {
        public Point3D GetPointUV(double u, double v)
        {
            // The central axis of the sphere points through world Y.
            // The U direction defines latitude and sweeps a full circle for 0 <= u <= 1.
            // The V direction defines longitude ans sweeps a half circle for 0 <= v <= 1.
            double ucos = Math.Cos(u * Math.PI * 2);
            double usin = Math.Sin(u * Math.PI * 2);
            double vcos = Math.Cos(v * Math.PI);
            double vsin = Math.Sin(v * Math.PI);
            return new Point3D(-usin * vsin, vcos, ucos * vsin);
        }
        public double RayTestDistance(Point3D origin, Vector3D direction)
        {
            double a = MathHelp.Dot(direction, direction);
            double b = 2 * MathHelp.Dot(origin, direction);
            double c = MathHelp.Dot(origin, origin) - SPHERE_RADIUS * SPHERE_RADIUS;
            // If the determinant is negative then there are no real roots and this will be NaN.
            double det = Math.Sqrt(b * b - 4 * a * c);
            // "a" cannot be negative so (worst case) these lambdas are +Inf.
            double den = 2 * a;
            double lambda1 = (-b - det) / den;
            double lambda2 = (-b + det) / den;
            double lambda_best = double.PositiveInfinity;
            if (lambda1 >= 0 && lambda1 < lambda_best) lambda_best = lambda1;
            if (lambda2 >= 0 && lambda2 < lambda_best) lambda_best = lambda2;
            return lambda_best;
        }
        public Vector3D RayTestNormal(Point3D origin, Vector3D direction)
        {
            double lambda = RayTestDistance(origin, direction);
            // Compute a vector from the center of the sphere to the intersection point.
            // This will a unit vector IFF the sphere has radius=1.
            // The caller is responsible for normalizing this as necessary.
            Point3D p = origin + lambda * direction;
            return new Vector3D(p.X, p.Y, p.Z);
        }
        const double SPHERE_RADIUS = 1;
    }
}