////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
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
            return BernsteinSum(Bernstein(u), Bernstein(v));
        }
        public Vector3D GetNormalUV(double u, double v)
        {
            // Computing the normal for a Bezier patch is a little involved.
            // First we compute the derivative of the U and V functions to find dP/dU and dP/dV.
            // With these two functions we can calculate the U and V tangents of the patch.
            // Cross these two vectors to compute the normal.
            Vector3D DPbyDU = BernsteinSum(BernsteinDerivative(u), Bernstein(v));
            Vector3D DPbyDV = BernsteinSum(Bernstein(u), BernsteinDerivative(v));
            return MathHelp.Normalized(MathHelp.Cross(DPbyDU, DPbyDV));
        }
        static double[] Bernstein(double u)
        {
            double nu = 1 - u;
            return new double[4] { nu * nu * nu, 3 * u * nu * nu, 3 * u * u * nu, u * u * u };
        }
        static double[] BernsteinDerivative(double u)
        {
            // bu[0] = (1-u)^3 = 1 - 3u + 3u^2 - u^3
            // bu[1] = 3u(1-u)^2 = 3u(1 - 2u + u^2) = 3u - 6u^2 + 3u^3
            // bu[2] = 3u^2(1-u) = 3u^2 - 3u^3
            // bu[3] = u^3
            // bu'[0] = -3 + 6u - 3u^2
            // bu'[1] = 3 - 12u + 9u^2
            // bu'[2] = 6u - 9u^2 = u(6 - 9u)
            // bu'[3] = 3u^2
            return new double[4] { -3 + 6 * u - 3 * u * u, 3 - 12 * u + 9 * u * u, 6 * u - 9 * u * u, 3 * u * u };
        }
        Vector3D BernsteinSum(double[] bu, double[] bv)
        {
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