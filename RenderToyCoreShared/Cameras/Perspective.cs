////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;

namespace RenderToy.Cameras
{
    public class Perspective : IProjection
    {
        public Matrix3D Projection
        {
            get
            {
                return CreateProjection(0.1, 10000.0, 60.0 * Math.PI / 180.0, 60.0 * Math.PI / 180.0);
            }
        }
        public static Matrix3D AspectCorrectFit(double width, double height)
        {
            double aspect = width / height;
            if (aspect > 1)
            {
                return MathHelp.CreateMatrixScale(1 / aspect, 1, 1);
            }
            else
            {
                return MathHelp.CreateMatrixScale(1, aspect, 1);
            }
        }
        public static Matrix3D CreateProjection(double near_plane, double far_plane, double fov_horiz, double fov_vert)
        {
            double w = 1 / Math.Tan(fov_horiz * 0.5);  // 1/tan(x) == cot(x)
            double h = 1 / Math.Tan(fov_vert * 0.5);   // 1/tan(x) == cot(x)
            double Q = far_plane / (far_plane - near_plane);
            Matrix3D result = new Matrix3D(
                w, 0, 0, 0,
                0, h, 0, 0,
                0, 0, Q, 1,
                0, 0, -Q * near_plane, 0
                );
            return result;
        }
    }
}