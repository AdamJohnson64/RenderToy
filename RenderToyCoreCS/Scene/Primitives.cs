﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;

namespace RenderToy
{
    /// <summary>
    /// This empty IPrimitive interface is only used to type-identify suitable primitive types.
    /// </summary>
    public interface IPrimitive
    {
    }
    public interface IParametricUV
    {
        /// <summary>
        /// Get a 3D point on this parametric surface.
        /// Parametric surfaces are only meaningfully defined in the range [0,1] in both U and V.
        /// </summary>
        /// <param name="u">The U location on the surface.</param>
        /// <param name="v">The V location on the surface.</param>
        /// <returns>A 3D point in object local space.</returns>
        Vector3D GetPointUV(double u, double v);
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
        Vector3D GetPointUVW(double u, double v, double w);
    }
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
    public class Cube : IPrimitive, IParametricUVW
    {
        public Vector3D GetPointUVW(double u, double v, double w)
        {
            return new Vector3D(-1 + u * 2, -1 + v * 2, -1 + w * 2);
        }
    }
    public class Cylinder : IPrimitive, IParametricUV
    {
        public Vector3D GetPointUV(double u, double v)
        {
            // The central axis of the sphere points through world Y.
            // The U direction defines latitude and sweeps a full circle for 0 <= u <= 1.
            // The V direction defines linear distance along Y.
            double ucos = Math.Cos(u * Math.PI * 2);
            double usin = Math.Sin(u * Math.PI * 2);
            return new Vector3D(-usin, -1 + v * 2, ucos);
        }
    }
    /// <summary>
    /// Plane in XZ.
    /// Note that for the purposes of parametric definitions this plane is bounded [-1,+1] in X and Z.
    /// The raytracer definition of this plane is infinite in the XZ plane.
    /// </summary>
    public class Plane : IPrimitive, IParametricUV
    {
        public Vector3D GetPointUV(double u, double v)
        {
            return new Vector3D(-1 + u * 2, 0, -1 + v * 2);
        }
    }
    /// <summary>
    /// Sphere of unit radius.
    /// The parametric definition of this sphere is oriented with the poles in Y.
    /// The "seam" of the sphere is deliberately behind the sphere in +Z.
    /// </summary>
    public class Sphere : IPrimitive, IParametricUV
    {
        public Vector3D GetPointUV(double u, double v)
        {
            // The central axis of the sphere points through world Y.
            // The U direction defines latitude and sweeps a full circle for 0 <= u <= 1.
            // The V direction defines longitude ans sweeps a half circle for 0 <= v <= 1.
            double ucos = Math.Cos(u * Math.PI * 2);
            double usin = Math.Sin(u * Math.PI * 2);
            double vcos = Math.Cos(v * Math.PI);
            double vsin = Math.Sin(v * Math.PI);
            return new Vector3D(-usin * vsin, vcos, ucos * vsin);
        }
    }
    /// <summary>
    /// Single triangle [0,0,0], [0,1,0], [1,0,0].
    /// </summary>
    public class Triangle : IPrimitive
    {
    }
}