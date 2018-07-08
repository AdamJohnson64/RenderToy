////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;

namespace RenderToy
{
    public static class OpenVRHelper
    {
        public static Matrix3D ConvertMatrix43(float[] matrix)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix[0]; m.M21 = matrix[1]; m.M31 = matrix[2]; m.M41 = matrix[3];
            m.M12 = matrix[4]; m.M22 = matrix[5]; m.M32 = matrix[6]; m.M42 = matrix[7];
            m.M13 = matrix[8]; m.M23 = matrix[9]; m.M33 = matrix[10]; m.M43 = matrix[11];
            m.M14 = 0; m.M24 = 0; m.M34 = 0; m.M44 = 1;
            return m;
        }
        public static Matrix3D ConvertMatrix44(float[] matrix)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix[0]; m.M21 = matrix[1]; m.M31 = matrix[2]; m.M41 = matrix[3];
            m.M12 = matrix[4]; m.M22 = matrix[5]; m.M32 = matrix[6]; m.M42 = matrix[7];
            m.M13 = matrix[8]; m.M23 = matrix[9]; m.M33 = matrix[10]; m.M43 = matrix[11];
            m.M14 = matrix[12]; m.M24 = matrix[13]; m.M34 = matrix[14]; m.M44 = matrix[15];
            return m;
        }
        public static Matrix3D ConvertMatrix44DX(float[] matrix, float scale)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix[0]; m.M21 = matrix[1]; m.M31 = -matrix[2]; m.M41 = matrix[3] * scale;
            m.M12 = matrix[4]; m.M22 = matrix[5]; m.M32 = -matrix[6]; m.M42 = matrix[7] * scale;
            m.M13 = matrix[8]; m.M23 = matrix[9]; m.M33 = -matrix[10]; m.M43 = matrix[11] * -scale; m.M44 = 1;
            return m;
        }
    }
}