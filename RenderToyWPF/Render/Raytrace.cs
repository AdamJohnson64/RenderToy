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
                for (int y = 0; y < buffer_height; ++y)
                {
                    byte* pRaster = (byte*)buffer_memory + y * buffer_stride;
                    for (int x = 0; x < buffer_width; ++x)
                    {
                        byte* pPixel = pRaster + x * 4;
                        Point4D v41 = new Point4D(-1.0 + ((x * 2.0) + 0.5) / buffer_width, 1.0 - ((y * 2.0) + 0.5) / buffer_height, 0, 1);
                        Point4D v42 = new Point4D(-1.0 + ((x * 2.0) + 0.5) / buffer_width, 1.0 - ((y * 2.0) + 0.5) / buffer_height, 1, 1);
                        v41 = inverse_mvp.Transform(v41);
                        v42 = inverse_mvp.Transform(v42);
                        v41 = MathHelp.Scale(v41, 1 / v41.W);
                        v42 = MathHelp.Scale(v42, 1 / v42.W);
                        Point3D v31 = new Point3D(v41.X, v41.Y, v41.Z);
                        Point3D v32 = new Point3D(v42.X, v42.Y, v42.Z);
                        double lambda_best = double.PositiveInfinity;
                        {
                            double lambda = double.PositiveInfinity;
                            if (IntersectPlane(v31, v32, ref lambda) && lambda >= 0 && lambda < lambda_best)
                            {
                                lambda_best = lambda;
                                pPixel[0] = 128;
                                pPixel[1] = 128;
                                pPixel[2] = 128;
                                pPixel[3] = 255;
                            }
                        }
                        {
                            double lambda1 = double.PositiveInfinity;
                            double lambda2 = double.PositiveInfinity;
                            if (IntersectSphere(v31, v32, 1.0, ref lambda1, ref lambda2))
                            {
                                if (lambda1 >= 0 && lambda1 < lambda_best)
                                {
                                    lambda_best = lambda1;
                                    pPixel[0] = 0;
                                    pPixel[1] = 0;
                                    pPixel[2] = 255;
                                    pPixel[3] = 255;
                                }
                                if (lambda2 >= 0 && lambda2 < lambda_best)
                                {
                                    lambda_best = lambda2;
                                    pPixel[0] = 0;
                                    pPixel[1] = 255;
                                    pPixel[2] = 0;
                                    pPixel[3] = 255;
                                }
                            }
                        }
                    }
                }
            }
        }
        static bool IntersectPlane(Point3D ray_origin, Point3D ray_direction, ref double lambda)
        {
            Point3D plane_normal = new Point3D(0, 1, 0);
            double plane_distance = 0;
            double det = MathHelp.Dot(plane_normal, ray_direction);
            if (det == 0) return false;
            lambda = (plane_distance - MathHelp.Dot(plane_normal, ray_origin)) / det;
            return true;
        }
        static bool IntersectSphere(Point3D ray_origin, Point3D ray_direction, double sphere_radius, ref double lambda1, ref double lambda2)
        {
            double a = MathHelp.Dot(ray_direction, ray_direction);
            double b = 2 * MathHelp.Dot(ray_origin, ray_direction);
            double c = MathHelp.Dot(ray_origin, ray_origin) - sphere_radius * sphere_radius;
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