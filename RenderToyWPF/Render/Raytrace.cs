using System;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    static class Raytrace
    {
        public static void DoRaytrace(Matrix3D inverse_mvp, int buffer_width, int buffer_height, IntPtr buffer_memory, int buffer_stride)
        {
            unsafe
            {
                // Define the scene.
                RaytraceObject[] objects = new RaytraceObject[]
                {
                    new RaytracePlane(Matrix3D.Identity, 0xff808080),
                    new RaytraceSphere(MathHelp.CreateTranslateMatrix(-2, 0, 0), 0xffff0000),
                    new RaytraceSphere(MathHelp.CreateTranslateMatrix(0, 0, 0), 0xff00ff00),
                    new RaytraceSphere(MathHelp.CreateTranslateMatrix(2, 0, 0), 0xff0000ff),
                };
                // Render the pixel buffer for the raytrace result.
                for (int y = 0; y < buffer_height; ++y)
                {
                    byte* pRaster = (byte*)buffer_memory + y * buffer_stride;
                    for (int x = 0; x < buffer_width; ++x)
                    {
                        // Compute an eye ray by unprojecting clip space rays via the inverse-mvp.
                        Point4D v41 = inverse_mvp.Transform(new Point4D(-1.0 + ((x * 2.0) + 0.5) / buffer_width, 1.0 - ((y * 2.0) + 0.5) / buffer_height, 0, 1));
                        Point4D v42 = inverse_mvp.Transform(new Point4D(-1.0 + ((x * 2.0) + 0.5) / buffer_width, 1.0 - ((y * 2.0) + 0.5) / buffer_height, 1, 1));
                        Point3D v31 = new Point3D(v41.X / v41.W, v41.Y / v41.W, v41.Z / v41.W);
                        Point3D v32 = new Point3D(v42.X / v42.W, v42.Y / v42.W, v42.Z / v42.W);
                        Ray ray = new Ray(v31, v32 - v31);
                        // Intersect test this ray against everything in the scene.
                        double found_lambda = double.PositiveInfinity;
                        uint found_color = 0;
                        foreach (var test in objects)
                        {
                            double lambda = test.Intersect(ray);
                            if (lambda >= 0 && lambda < found_lambda)
                            {
                                found_lambda = lambda;
                                found_color = test.color;
                            }
                        }
                        // If we hit something then output the pixel.
                        if (found_lambda != double.PositiveInfinity)
                        {
                            byte* pPixel = pRaster + x * 4;
                            *(uint*)pPixel = found_color;
                        }
                    }
                }
            }
        }
    }
    struct Ray
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
    /// <summary>
    /// Abstract raytrace object.
    /// </summary>
    abstract class RaytraceObject
    {
        protected RaytraceObject(Matrix3D transform, uint color)
        {
            this.transform = transform;
            this.transform_inverse = MathHelp.Invert(transform);
            this.color = color;
        }
        /// <summary>
        /// Intersect this object in transformed space.
        /// The supplied ray will be inverse transformed into the space of the object before testing.
        /// </summary>
        /// <param name="ray_origin">The world space starting point of the ray.</param>
        /// <param name="ray_direction">The world space ending point of the ray.</param>
        /// <returns>The positive distance along the ray direction to intersection (or +inf for no intersection).</returns>
        public double Intersect(Ray ray)
        {
            return IntersectLocal(ray.Transform(transform_inverse));
        }
        /// <summary>
        /// Object specific ray intersection test.
        /// </summary>
        /// <param name="ray_origin">The object space starting point of the ray.</param>
        /// <param name="ray_direction">The object space ending point of the ray.</param>
        /// <returns>The positive distance along the ray direction to intersection (or +inf for no intersection).</returns>
        protected abstract double IntersectLocal(Ray ray);
        private Matrix3D transform;
        private Matrix3D transform_inverse;
        public uint color;
    }
    /// <summary>
    /// Raytracer plane (in XZ).
    /// </summary>
    class RaytracePlane : RaytraceObject
    {
        public RaytracePlane(Matrix3D transform, uint color) : base(transform, color)
        {
        }
        protected override double IntersectLocal(Ray ray)
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
    /// Raytracer unit-radius sphere.
    /// </summary>
    class RaytraceSphere : RaytraceObject
    {
        public RaytraceSphere(Matrix3D transform, uint color) : base(transform, color)
        {
        }
        protected override double IntersectLocal(Ray ray)
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