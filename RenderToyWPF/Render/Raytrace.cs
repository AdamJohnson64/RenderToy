////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    static class Raytrace
    {
        public static void DoRaytrace(Scene scene, Matrix3D inverse_mvp, int buffer_width, int buffer_height, IntPtr buffer_memory, int buffer_stride)
        {
            // Define the scene.
            RaytraceObject[] objects =
                TransformedObject.Enumerate(scene)
                .Where(x => x.Node.Primitive is IRayTest)
                .Select(x => new RaytraceObject(x.Transform, (IRayTest)x.Node.Primitive, ColorToARGB(x.Node.WireColor), x.Node.material))
                .ToArray();
            // Render the pixel buffer for the raytrace result.
            Point3D found_color = new Point3D(0, 0, 0);
            for (int y = 0; y < buffer_height; ++y)
            {
                for (int x = 0; x < buffer_width; ++x)
                {
                    // Compute an eye ray by unprojecting clip space rays via the inverse-mvp.
                    Point4D v41 = inverse_mvp.Transform(new Point4D(-1.0 + ((x * 2.0) + 0.5) / buffer_width, 1.0 - ((y * 2.0) + 0.5) / buffer_height, 0, 1));
                    Point4D v42 = inverse_mvp.Transform(new Point4D(-1.0 + ((x * 2.0) + 0.5) / buffer_width, 1.0 - ((y * 2.0) + 0.5) / buffer_height, 1, 1));
                    Point3D ray_origin = new Point3D(v41.X / v41.W, v41.Y / v41.W, v41.Z / v41.W);
                    Vector3D ray_direction = new Point3D(v42.X / v42.W, v42.Y / v42.W, v42.Z / v42.W) - ray_origin;
                    if (!RayColor(objects, ray_origin, ray_direction, ref found_color)) continue;
                    uint color = Point3ToARGB(found_color);
                    unsafe
                    {
                        byte* pRaster = (byte*)buffer_memory + y * buffer_stride;
                        byte* pPixel = pRaster + x * 4;
                        *(uint*)pPixel = color;
                    }
                }
            }
        }
        static bool RayColor(RaytraceObject[] objects, Point3D ray_origin, Vector3D ray_direction, ref Point3D output_color)
        {
            // Intersect test this ray against everything in the scene.
            double found_lambda = double.PositiveInfinity;
            RaytraceObject found_object = null;
            RayIntersectFirst(objects, ray_origin, ray_direction, ref found_lambda, ref found_object);
            // If we hit something then output the pixel.
            if (double.IsPositiveInfinity(found_lambda)) return false;
            // World Vector - The world space position of intersection.
            Point3D vector_world = ray_origin + found_lambda * ray_direction;
            // Normal Vector - The world space normal of the intersection point.
            Vector3D vector_normal = found_object.RayTestNormal(ray_origin, ray_direction);
            vector_normal.Normalize();
            // View Vector - The world space direction we are looking.
            Vector3D vector_view = -ray_direction;
            vector_view.Normalize();
            // Light Vector - The world space direction toward the light.
            Vector3D vector_light = new Point3D(10, 10, -10) - vector_world;
            vector_light.Normalize();
            // Reflection Vector - The world space reflected view about the normal.
            Vector3D vector_reflect = -vector_view + 2 * MathHelp.Dot(vector_normal, vector_view) * vector_normal;
            // Shadow test.
            Point3D shadow_origin = ray_origin + found_lambda * ray_direction + 0.0001 * vector_normal;
            double shadow_multiplier = RayIntersectAny(objects, shadow_origin, vector_light) ? 0.5 : 1.0;
            // Calculate material color.
            Color color_material = found_object.material.MaterialCompute(ray_origin, ray_direction, found_lambda);
            // Lighting scales.
            double lambert = Math.Max(0, Math.Min(1, MathHelp.Dot(vector_normal, vector_light) * shadow_multiplier));
            double specular = Math.Pow(Math.Max(0, Math.Min(1, MathHelp.Dot(vector_reflect, vector_light))), 10);
            // Compute the overall color of the surface.
            output_color.X = lambert * color_material.R / 255.0 + specular;
            output_color.Y = lambert * color_material.G / 255.0 + specular;
            output_color.Z = lambert * color_material.B / 255.0 + specular;
            return true;
        }
        static bool RayIntersectAny(RaytraceObject[] objects, Point3D ray_origin, Vector3D ray_direction)
        {
            foreach (var test in objects)
            {
                double lambda = test.RayTestDistance(ray_origin, ray_direction);
                if (lambda >= 0 && !double.IsPositiveInfinity(lambda)) return true;
            }
            return false;
        }
        static void RayIntersectFirst(RaytraceObject[] objects, Point3D ray_origin, Vector3D ray_direction, ref double output_lambda, ref RaytraceObject output_object)
        {
            double found_lambda = double.PositiveInfinity;
            RaytraceObject found_object = null;
            foreach (var test in objects)
            {
                double lambda = test.RayTestDistance(ray_origin, ray_direction);
                if (lambda >= 0 && lambda < found_lambda)
                {
                    found_object = test;
                    found_lambda = lambda;
                }
            }
            output_lambda = found_lambda;
            output_object = found_object;
        }
        static uint ColorToARGB(Color color)
        {
            return
                ((uint)color.A << 24) |
                ((uint)color.R << 16) |
                ((uint)color.G << 8) |
                ((uint)color.B << 0);
        }
        static uint Point3ToARGB(Point3D color)
        {
            // Clamp the color components so we don't oversaturate.
            double r = Math.Max(0, Math.Min(1, color.X));
            double g = Math.Max(0, Math.Min(1, color.Y));
            double b = Math.Max(0, Math.Min(1, color.Z));
            return
                (0xffU << 24) |             // Alpha (MSB)
                ((uint)(r * 0xffU) << 16) | // Red
                ((uint)(g * 0xffU) << 8) |  // Green
                ((uint)(b * 0xffU) << 0);   // Blue (LSB)
        }
    }
    /// <summary>
    /// Abstract raytrace object.
    /// </summary>
    class RaytraceObject
    {
        public RaytraceObject(Matrix3D transform, IRayTest primitive, uint color, IMaterial material)
        {
            this.transform = transform;
            this.transform_inverse = MathHelp.Invert(transform);
            this.primitive = primitive;
            this.color = color;
            this.material = material;
        }
        /// <summary>
        /// Intersect this object in transformed space.
        /// The supplied ray will be inverse transformed into the space of the object before testing.
        /// </summary>
        /// <param name="ray">The world space ray to test.</param>
        /// <returns>The positive distance along the ray direction to the first intersection (or +inf for no intersection was found).</returns>
        public double RayTestDistance(Point3D origin, Vector3D direction)
        {
            return primitive.RayTestDistance(
                transform_inverse.Transform(origin),
                transform_inverse.Transform(direction));
        }
        public Vector3D RayTestNormal(Point3D origin, Vector3D direction)
        {
            Vector3D normal = transform.Transform(
                primitive.RayTestNormal(
                    transform_inverse.Transform(origin),
                    transform_inverse.Transform(direction)));
            return normal;
        }
        private Matrix3D transform;
        private Matrix3D transform_inverse;
        public IRayTest primitive;
        public uint color;
        public IMaterial material;
    }
}