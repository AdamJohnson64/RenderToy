////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;

namespace RenderToy.Primitives
{
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
}