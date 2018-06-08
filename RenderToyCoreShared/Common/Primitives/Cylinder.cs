////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;

namespace RenderToy.Primitives
{
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
        public Vector3D GetNormalUV(double u, double v)
        {
            // Remove the Y component for the normal.
            double ucos = Math.Cos(u * Math.PI * 2);
            double usin = Math.Sin(u * Math.PI * 2);
            return new Vector3D(-usin, 0, ucos);
        }
        public Vector3D GetTangentUV(double u, double v)
        {
            return new Vector3D(0, 0, 0);
        }
        public Vector3D GetBitangentUV(double u, double v)
        {
            return new Vector3D(0, 0, 0);
        }
    }
}