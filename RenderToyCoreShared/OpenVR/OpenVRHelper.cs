////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#if OPENVR_INSTALLED
using RenderToy.Math;

namespace RenderToy
{
    public static class OpenVRHelper
    {
        public static Matrix3D ConvertMatrix43(HmdMatrix34 matrix)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix.M11; m.M21 = matrix.M21; m.M31 = matrix.M31; m.M41 = matrix.M41;
            m.M12 = matrix.M12; m.M22 = matrix.M22; m.M32 = matrix.M32; m.M42 = matrix.M42;
            m.M13 = matrix.M13; m.M23 = matrix.M23; m.M33 = matrix.M33; m.M43 = matrix.M43;
            m.M14 = 0; m.M24 = 0; m.M34 = 0; m.M44 = 1;
            return m;
        }
        public static Matrix3D ConvertMatrix44(HmdMatrix44 matrix)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix.M11; m.M21 = matrix.M21; m.M31 = matrix.M31; m.M41 = matrix.M41;
            m.M12 = matrix.M12; m.M22 = matrix.M22; m.M32 = matrix.M32; m.M42 = matrix.M42;
            m.M13 = matrix.M13; m.M23 = matrix.M23; m.M33 = matrix.M33; m.M43 = matrix.M43;
            m.M14 = matrix.M14; m.M24 = matrix.M24; m.M34 = matrix.M34; m.M44 = matrix.M44;
            return m;
        }
        public static Matrix3D GetEyeToHeadTransform(Eye eEye)
        {
            return ConvertMatrix43(OpenVR.GetEyeToHeadTransform(eEye));
        }
        public static Matrix3D GetProjectionMatrix(Eye eEye, float fNear, float fFar)
        {
            return ConvertMatrix44(OpenVR.GetProjectionMatrix(eEye, fNear, fFar));
        }
    }
}
#endif // OPENVR_INSTALLED