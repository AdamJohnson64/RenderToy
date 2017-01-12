////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static class MathHelp
    {
        public static Matrix3D CreateLookAt(Point3D eye, Point3D at, Vector3D up)
        {
            Vector3D lfz = MathHelp.Normalized(at - eye);
            Vector3D lfx = MathHelp.Normalized(MathHelp.Cross(up, lfz));
            Vector3D lfy = MathHelp.Normalized(MathHelp.Cross(lfz, lfx));
            return new Matrix3D(
                lfx.X, lfx.Y, lfx.Z, 0,
                lfy.X, lfy.Y, lfy.Z, 0,
                lfz.X, lfz.Y, lfz.Z, 0,
                eye.X, eye.Y, eye.Z, 1);
        }
        public static Matrix3D CreateScaleMatrix(double x, double y, double z)
        {
            return new Matrix3D(
                x, 0, 0, 0,
                0, y, 0, 0,
                0, 0, z, 0,
                0, 0, 0, 1);
        }
        public static Matrix3D CreateTranslateMatrix(double x, double y, double z)
        {
            return new Matrix3D(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                x, y, z, 1);
        }
        public static Point3D Add(Point3D a, Point3D b)
        {
            return new Point3D(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        }
        public static Vector3D Cross(Vector3D a, Vector3D b)
        {
            return new Vector3D(
                a.Y * b.Z - a.Z * b.Y,
                a.Z * b.X - a.X * b.Z,
                a.X * b.Y - a.Y * b.X);
        }
        public static double Dot(Point3D a, Point3D b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }
        public static double Dot(Point3D a, Vector3D b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }
        public static double Dot(Vector3D a, Point3D b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }
        public static double Dot(Vector3D a, Vector3D b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }
        public static double Dot(Point4D a, Point4D b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
        }
        public static Matrix3D Invert(Matrix3D m)
        {
            m.Invert();
            return m;
        }
        public static Vector3D Normalized(Vector3D v)
        {
            v.Normalize();
            return v;
        }
        public static Point3D Scale(Point3D v, double a)
        {
            return new Point3D(v.X * a, v.Y * a, v.Z * a);
        }
        public static Point4D Scale(Point4D v, double a)
        {
            return new Point4D(v.X * a, v.Y * a, v.Z * a, v.W * a);
        }
        public static Matrix3D ToMatrix(Quaternion q)
        {
            QuaternionRotation3D quatrot3 = new QuaternionRotation3D(q);
            RotateTransform3D rottran3 = new RotateTransform3D(quatrot3);
            return rottran3.Value;
        }
        public static Matrix3D ToMatrix(Vector3D v)
        {
            TranslateTransform3D trantran3 = new TranslateTransform3D(v);
            return trantran3.Value;
        }
    }
}