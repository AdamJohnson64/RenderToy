﻿using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static class MathHelp
    {
        public static double Dot(Point4D a, Point4D b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
        }
        public static Matrix3D Invert(Matrix3D m)
        {
            m.Invert();
            return m;
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