////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;

namespace RenderToy.Primitives
{
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
}