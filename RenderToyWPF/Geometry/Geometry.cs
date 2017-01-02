using System;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public interface IParametricUV
    {
        Point3D GetPointUV(double u, double v);
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
            return new Point3D(-usin, v, ucos);
        }
    }
    public class Sphere : IParametricUV
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
    }
    public class BezierPatch : IParametricUV
    {
        public Point3D GetPointUV(double u, double v)
        {
            // Define the hull for the patch.
            const double h = 0.5;
            Point3D[] hull = new Point3D[16]
            {
                new Point3D(-1, 0, -1), new Point3D(-h, 0, -1), new Point3D(+h, 0, -1), new Point3D(+1, 0, -1),
                new Point3D(-1, 0, -h), new Point3D(-h, 1, -h), new Point3D(+h, 0, -h), new Point3D(+1, 0, -h),
                new Point3D(-1, 0, +h), new Point3D(-h, 0, +h), new Point3D(+h, 0, +h), new Point3D(+1, 0, +h),
                new Point3D(-1, 0, +1), new Point3D(-h, 0, +1), new Point3D(+h, 0, +1), new Point3D(+1, 0, +1),
            };
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
    }
}