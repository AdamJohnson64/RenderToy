////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;

namespace RenderToy.Primitives
{
    public class Cube : IPrimitive, IParametricUVW
    {
        public Vector3D GetPointUVW(double u, double v, double w)
        {
            return new Vector3D(-1 + u * 2, -1 + v * 2, -1 + w * 2);
        }
    }
}