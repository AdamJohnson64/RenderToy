////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;

namespace RenderToy.Primitives
{
    /// <summary>
    /// Sphere of unit radius.
    /// The parametric definition of this sphere is oriented with the poles in Y.
    /// The "seam" of the sphere is deliberately behind the sphere in +Z.
    /// </summary>
    public class Sphere : IPrimitive, IParametricUV
    {
        public static Sphere Default = new Sphere();
        private Sphere()
        {
        }
        public Vector3D GetPointUV(double u, double v)
        {
            // The central axis of the sphere points through world Y.
            // The U direction defines latitude and sweeps a full circle for 0 <= u <= 1.
            // The V direction defines longitude ans sweeps a half circle for 0 <= v <= 1.
            double ucos = System.Math.Cos(u * System.Math.PI * 2);
            double usin = System.Math.Sin(u * System.Math.PI * 2);
            double vcos = System.Math.Cos(v * System.Math.PI);
            double vsin = System.Math.Sin(v * System.Math.PI);
            return new Vector3D(-usin * vsin, vcos, ucos * vsin);
        }
        public Vector3D GetNormalUV(double u, double v)
        {
            // For a unit sphere the normal is the same as the surface point.
            return GetPointUV(u, v);
        }
        public Vector3D GetTangentUV(double u, double v)
        {
            double ucos = System.Math.Cos(u * System.Math.PI * 2);
            double usin = System.Math.Sin(u * System.Math.PI * 2);
            double vcos = System.Math.Cos(v * System.Math.PI);
            double vsin = System.Math.Sin(v * System.Math.PI);
            return MathHelp.Normalized(new Vector3D(-ucos, 0, -usin));
        }
        public Vector3D GetBitangentUV(double u, double v)
        {
            double ucos = System.Math.Cos(u * System.Math.PI * 2);
            double usin = System.Math.Sin(u * System.Math.PI * 2);
            double vcos = System.Math.Cos(v * System.Math.PI);
            double vsin = System.Math.Sin(v * System.Math.PI);
            return MathHelp.Normalized(new Vector3D(-usin * vcos, -vsin, ucos * vcos));
        }
    }
}