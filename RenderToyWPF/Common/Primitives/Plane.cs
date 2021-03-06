﻿using RenderToy.Math;

namespace RenderToy.Primitives
{
    /// <summary>
    /// Plane in XZ.
    /// Note that for the purposes of parametric definitions this plane is bounded [-1,+1] in X and Z.
    /// The raytracer definition of this plane is infinite in the XZ plane.
    /// </summary>
    public class Plane : IPrimitive, IParametricUV
    {
        public static Plane Default = new Plane();
        private Plane()
        {
        }
        public Vector3D GetPointUV(double u, double v)
        {
            return new Vector3D(-1 + u * 2, 0, 1 - v * 2);
        }
        public Vector3D GetNormalUV(double u, double v)
        {
            return new Vector3D(0, 1, 0);
        }
        public Vector3D GetTangentUV(double u, double v)
        {
            return new Vector3D(1, 0, 0);
        }
        public Vector3D GetBitangentUV(double u, double v)
        {
            return new Vector3D(0, 0, -1);
        }
    }
}