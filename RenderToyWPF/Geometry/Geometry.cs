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
    public struct Ray
    {
        public Ray(Point3D origin, Vector3D direction)
        {
            O = origin;
            D = direction;
        }
        public Ray Transform(Matrix3D transform)
        {
            return new Ray(
                transform.Transform(O),
                transform.Transform(D));
        }
        public Point3D O;
        public Vector3D D;
    }
    interface IRayTest
    {
        /// <summary>
        /// Object specific ray intersection test.
        /// </summary>
        /// <param name="ray">A ray origin and direction pair.</param>
        /// <returns>The positive distance along the ray direction to the first intersection (or +inf for no intersection was found).</returns>
        double RayTest(Ray ray);
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
                new Point3D(-1, 0, -h), new Point3D(-h, 1, -h), new Point3D(+h, 1, -h), new Point3D(+1, 0, -h),
                new Point3D(-1, 0, +h), new Point3D(-h, 1, +h), new Point3D(+h, 1, +h), new Point3D(+1, 0, +h),
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
                    acc = MathHelp.Add(acc, MathHelp.Scale(hull[i + j * 4], bu[i] * bv[j]));
                }
            }
            return acc;
        }
        Point3D[] hull = null;
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
        public double RayTest(Ray ray)
        {
            double lambda_best = double.PositiveInfinity;
            double lambda = double.PositiveInfinity;
            if (IntersectPlane(ray, new Vector3D(0, 1, 0), 0, ref lambda) && lambda >= 0 && lambda < lambda_best)
            {
                lambda_best = lambda;
            }
            return lambda_best;
        }
        private static bool IntersectPlane(Ray ray, Vector3D plane_normal, double plane_distance, ref double lambda)
        {
            double det = MathHelp.Dot(plane_normal, ray.D);
            if (det == 0) return false;
            lambda = (plane_distance - MathHelp.Dot(plane_normal, ray.O)) / det;
            return true;
        }
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
        public double RayTest(Ray ray)
        {
            double lambda_best = double.PositiveInfinity;
            double lambda1 = double.PositiveInfinity;
            double lambda2 = double.PositiveInfinity;
            if (IntersectSphere(ray, 1.0, ref lambda1, ref lambda2))
            {
                if (lambda1 >= 0 && lambda1 < lambda_best)
                {
                    lambda_best = lambda1;
                }
                if (lambda2 >= 0 && lambda2 < lambda_best)
                {
                    lambda_best = lambda2;
                }
            }
            return lambda_best;
        }
        private static bool IntersectSphere(Ray ray, double sphere_radius, ref double lambda1, ref double lambda2)
        {
            double a = MathHelp.Dot(ray.D, ray.D);
            double b = 2 * MathHelp.Dot(ray.O, ray.D);
            double c = MathHelp.Dot(ray.O, ray.O) - sphere_radius * sphere_radius;
            double det = b * b - 4 * a * c;
            if (det <= 0) return false;
            det = Math.Sqrt(det);
            double den = 2 * a;
            lambda1 = (-b - det) / den;
            lambda2 = (-b + det) / den;
            return true;
        }
    }
}