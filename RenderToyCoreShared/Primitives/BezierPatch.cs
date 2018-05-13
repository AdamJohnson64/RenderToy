////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;

namespace RenderToy.Primitives
{
    public class BezierPatch : IPrimitive, IParametricUV
    {
        public BezierPatch()
        {
            // Define the hull for the patch.
            const double h = 0.5;
            hull = new Vector3D[16]
            {
                new Vector3D(-1, 0, -1), new Vector3D(-h, 0, -1), new Vector3D(+h, 0, -1), new Vector3D(+1, 0, -1),
                new Vector3D(-1, 0, -h), new Vector3D(-h, 4, -h), new Vector3D(+h, -4, -h), new Vector3D(+1, 0, -h),
                new Vector3D(-1, 0, +h), new Vector3D(-h, -4, +h), new Vector3D(+h, 4, +h), new Vector3D(+1, 0, +h),
                new Vector3D(-1, 0, +1), new Vector3D(-h, 0, +1), new Vector3D(+h, 0, +1), new Vector3D(+1, 0, +1),
            };
        }
        public BezierPatch(Vector3D[] hull)
        {
            this.hull = hull;
        }
        public Vector3D GetPointUV(double u, double v)
        {
            // The Bernstein polynomial factors.
            double nu = 1 - u;
            double[] bu = new double[4] { nu * nu * nu, 3 * u * nu * nu, 3 * u * u * nu, u * u * u };
            double nv = 1 - v;
            double[] bv = new double[4] { nv * nv * nv, 3 * v * nv * nv, 3 * v * v * nv, v * v * v };
            // Compute the UV point.
            Vector3D acc = new Vector3D(0, 0, 0);
            for (int j = 0; j < 4; ++j)
            {
                for (int i = 0; i < 4; ++i)
                {
                    acc = MathHelp.Add(acc, MathHelp.Multiply(hull[i + j * 4], bu[i] * bv[j]));
                }
            }
            return acc;
        }
        Vector3D[] hull = null;
    }
}