////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

#define USE_WPF

using System;
using System.Diagnostics;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static partial class MathHelp
    {
        public static Point3D Add(Point3D a, Point3D b) { return new Point3D(a.X + b.X, a.Y + b.Y, a.Z + b.Z); }
        public static Point3D Add(Point3D a, Vector3D b) { return new Point3D(a.X + b.X, a.Y + b.Y, a.Z + b.Z); }
        public static Point3D Add(Vector3D a, Point3D b) { return new Point3D(a.X + b.X, a.Y + b.Y, a.Z + b.Z); }
        public static Point4D Add(Point4D a, Point4D b) { return new Point4D(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W); }
        public static Vector3D Add(Vector3D a, Vector3D b) { return new Vector3D(a.X + b.X, a.Y + b.Y, a.Z + b.Z); }
        public static Vector3D Cross(Vector3D a, Vector3D b) { return new Vector3D(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X); }
        public static double Dot(Point3D a, Point3D b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
        public static double Dot(Point3D a, Vector3D b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
        public static double Dot(Vector3D a, Point3D b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
        public static double Dot(Vector3D a, Vector3D b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
        public static double Dot(Point4D a, Point4D b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W; }
        public static double Length(Point3D p) { return Math.Sqrt(Dot(p, p)); }
        public static double Length(Point4D p) { return Math.Sqrt(Dot(p, p)); }
        public static double Length(Vector3D v) { return Math.Sqrt(Dot(v, v)); }
        public static Point3D Multiply(Point3D a, double b) { return new Point3D(a.X * b, a.Y * b, a.Z * b); }
        public static Point3D Multiply(double a, Point3D b) { return Multiply(b, a); }
        public static Point4D Multiply(Point4D a, double b) { return new Point4D(a.X * b, a.Y * b, a.Z * b, a.W * b); }
        public static Point4D Multiply(double a, Point4D b) { return Multiply(b, a); }
        public static Vector3D Multiply(Vector3D a, double b) { return new Vector3D(a.X * b, a.Y * b, a.Z * b); }
        public static Vector3D Multiply(double a, Vector3D b) { return Multiply(b, a); }
        public static Point3D Normalized(Point3D v) { return Multiply(v, 1 / Length(v)); }
        public static Point4D Normalized(Point4D v) { return Multiply(v, 1 / Length(v)); }
        public static Vector3D Normalized(Vector3D v) { return v * (1 / Length(v)); }
        public static Vector3D Subtract(Point3D a, Point3D b) { return new Vector3D(a.X - b.X, a.Y - b.Y, a.Z - b.Z); }
        public static Point4D Subtract(Point4D a, Point4D b) { return new Point4D(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W); }
        public static Matrix3D CreateScaleMatrix(double x, double y, double z) { return new Matrix3D(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1); }
        public static Matrix3D CreateTranslateMatrix(double x, double y, double z) { return new Matrix3D(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1); }
        public static Matrix3D CreateLookAt(Point3D eye, Point3D at, Vector3D up)
        {
            var lfz = MathHelp.Normalized(at - eye);
            var lfx = MathHelp.Normalized(MathHelp.Cross(up, lfz));
            var lfy = MathHelp.Normalized(MathHelp.Cross(lfz, lfx));
            return new Matrix3D(
                lfx.X, lfx.Y, lfx.Z, 0,
                lfy.X, lfy.Y, lfy.Z, 0,
                lfz.X, lfz.Y, lfz.Z, 0,
                eye.X, eye.Y, eye.Z, 1);
        }
        public static System.Windows.Media.Media3D.Matrix3D Invert(System.Windows.Media.Media3D.Matrix3D m)
        {
            m.Invert();
            return m;
        }
        public static System.Windows.Media.Media3D.Matrix3D ToMatrix(System.Windows.Media.Media3D.Quaternion q)
        {
            var quatrot3 = new QuaternionRotation3D(q);
            var rottran3 = new RotateTransform3D(quatrot3);
            return rottran3.Value;
        }
    }
#if !USE_WPF
    public static partial class MathHelp
    {
        public static Matrix3D Invert(RenderToy.Matrix3D m)
        {
            var mw = new System.Windows.Media.Media3D.Matrix3D(
                m.M11, m.M12, m.M13, m.M14,
                m.M21, m.M22, m.M23, m.M24,
                m.M31, m.M32, m.M33, m.M34,
                m.M41, m.M42, m.M43, m.M44);
            mw.Invert();
            return new Matrix3D(
                mw.M11, mw.M12, mw.M13, mw.M14,
                mw.M21, mw.M22, mw.M23, mw.M24,
                mw.M31, mw.M32, mw.M33, mw.M34,
                mw.OffsetX, mw.OffsetY, mw.OffsetZ, mw.M44);
        }
        public static Matrix3D ToMatrix(RenderToy.Quaternion q)
        {
            double m11 = 1 - 2 * q.Y * q.Y - 2 * q.Z * q.Z;
            double m12 = 2 * (q.X * q.Y + q.Z * q.W);
            double m13 = 2 * (q.X * q.Z - q.Y * q.W);
            double m21 = 2 * (q.X * q.Y - q.Z * q.W);
            double m22 = 1 - 2 * q.X * q.X - 2 * q.Z * q.Z;
            double m23 = 2 * (q.Y * q.Z + q.X * q.W);
            double m31 = 2 * (q.X * q.Z + q.Y * q.W);
            double m32 = 2 * (q.Y * q.Z - q.X * q.W);
            double m33 = 1 - 2 * q.X * q.X - 2 * q.Y * q.Y;
            return new Matrix3D(m11, m12, m13, 0, m21, m22, m23, 0, m31, m32, m33, 0, 0, 0, 0, 1);
        }
    }
    public struct Vector3D
    {
        public double X, Y, Z;
        public Vector3D(double x, double y, double z) { X = x; Y = y; Z = z; }
        public static Vector3D operator -(Vector3D a) { return new Vector3D(-a.X, -a.Y, -a.Z); }
        public static Vector3D operator +(Vector3D a, Vector3D b) { return MathHelp.Add(a, b); }
        public static Vector3D operator *(Vector3D a, double b) { return MathHelp.Multiply(a, b); }
        public static Vector3D operator *(double a, Vector3D b) { return MathHelp.Multiply(a, b); }
    }
    public struct Point3D
    {
        public double X, Y, Z;
        public Point3D(double x, double y, double z) { X = x; Y = y; Z = z; }
        public static Point3D operator -(Point3D a) { return new Point3D(-a.X, -a.Y, -a.Z); }
        public static Point3D operator +(Point3D a, Point3D b) { return MathHelp.Add(a,b); }
        public static Point3D operator +(Point3D a, Vector3D b) { return MathHelp.Add(a, b); }
        public static Point3D operator +(Vector3D a, Point3D b) { return MathHelp.Add(a, b); }
        public static Vector3D operator -(Point3D a, Point3D b) { return MathHelp.Subtract(a, b); }
    }
    public struct Point4D
    {
        public double X, Y, Z, W;
        public Point4D(double x, double y, double z, double w) { X = x;  Y = y;  Z = z; W = w; }
        public static Point4D operator -(Point4D a) { return new Point4D(-a.X, -a.Y, -a.Z, -a.W); }
        public static Point4D operator +(Point4D a, Point4D b) { return MathHelp.Add(a, b); }
        public static Point4D operator -(Point4D a, Point4D b) { return MathHelp.Subtract(a, b); }
    }
    public struct Matrix3D
    {
        public double M11, M12, M13, M14;
        public double M21, M22, M23, M24;
        public double M31, M32, M33, M34;
        public double M41, M42, M43, M44;
        public Matrix3D(
            double m11, double m12, double m13, double m14,
            double m21, double m22, double m23, double m24,
            double m31, double m32, double m33, double m34,
            double m41, double m42, double m43, double m44)
        {
            M11 = m11; M12 = m12; M13 = m13; M14 = m14;
            M21 = m21; M22 = m22; M23 = m23; M24 = m24;
            M31 = m31; M32 = m32; M33 = m33; M34 = m34;
            M41 = m41; M42 = m42; M43 = m43; M44 = m44;
        }
        public Matrix3D(double[] m) : this(
            m[00], m[01], m[02], m[03],
            m[04], m[05], m[06], m[07],
            m[08], m[09], m[10], m[11],
            m[12], m[13], m[14], m[15])
        {
        }
        public static readonly Matrix3D Identity = new Matrix3D(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        public static Matrix3D operator*(Matrix3D a, Matrix3D b)
        {
            double[] m = new double[16];
            for (int row = 0; row < 4; ++row)
            {
                for (int col = 0; col < 4; ++col)
                {
                    m[col + 4 * row] = MathHelp.Dot(a.GetRow(row), b.GetColumn(col));
                }
            }
            return new Matrix3D(m);
        }
        public Point3D Transform(Point3D p)
        {
            var p4 = Transform(new Point4D(p.X, p.Y, p.Z, 1));
            return new Point3D(p4.X, p4.Y, p4.Z);
        }
        public Point4D Transform(Point4D p)
        {
            return new Point4D(
                MathHelp.Dot(GetColumn(0), p),
                MathHelp.Dot(GetColumn(1), p),
                MathHelp.Dot(GetColumn(2), p),
                MathHelp.Dot(GetColumn(3), p));
        }
        public Vector3D Transform(Vector3D p)
        {
            var p4 = Transform(new Point4D(p.X, p.Y, p.Z, 0));
            return new Vector3D(p4.X, p4.Y, p4.Z);
        }
        public Point4D GetColumn(int col)
        {
            switch (col)
            {
                case 0: return new Point4D(M11, M21, M31, M41);
                case 1: return new Point4D(M12, M22, M32, M42);
                case 2: return new Point4D(M13, M23, M33, M43);
                case 3: return new Point4D(M14, M24, M34, M44);
                default: throw new System.IndexOutOfRangeException();
            }
        }
        Point4D GetRow(int row)
        {
            switch (row)
            {
                case 0: return new Point4D(M11, M12, M13, M14);
                case 1: return new Point4D(M21, M22, M23, M24);
                case 2: return new Point4D(M31, M32, M33, M34);
                case 3: return new Point4D(M41, M42, M43, M44);
                default: throw new System.IndexOutOfRangeException();
            }
        }
    }
    public struct Quaternion
    {
        public double X, Y, Z, W;
        public Quaternion(double x, double y, double z, double w) { X = x; Y = y; Z = z; W = w; }
        public Quaternion(Vector3D axis, double angle)
        {
            double radians_over_2 = (angle * Math.PI / 180) / 2;
            double sin = Math.Sin(radians_over_2);
            Vector3D normalized_axis = MathHelp.Normalized(axis);
            X = normalized_axis.X * sin;
            Y = normalized_axis.Y * sin;
            Z = normalized_axis.Z * sin;
            W = Math.Cos(radians_over_2);
        }
        public static Quaternion operator *(Quaternion q, Quaternion r)
        {
            double t0 = r.W * q.W - r.X * q.X - r.Y * q.Y - r.Z * q.Z;
            double t1 = r.W * q.X + r.X * q.W - r.Y * q.Z + r.Z * q.Y;
            double t2 = r.W * q.Y + r.X * q.Z + r.Y * q.W - r.Z * q.X;
            double t3 = r.W * q.Z - r.X * q.Y + r.Y * q.X + r.Z * q.W;
            double ln = 1 / Math.Sqrt(t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3);
            t0 *= ln; t1 *= ln; t2 *= ln; t3 *= ln;
            return new Quaternion(t1, t2, t3, t0);
        }
    }
#endif
    namespace Test
    {
        public static class MathHelp
        {
            public static void Run()
            {
                // Matrix3D Multiply.
                {
                    var scale = new Matrix3D(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1);
                    var transform = new Matrix3D(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1);
                    var result = scale * transform;
                    Debug.Assert(result.M11 == 2); Debug.Assert(result.M12 == 0); Debug.Assert(result.M13 == 0); Debug.Assert(result.M14 == 0);
                    Debug.Assert(result.M21 == 0); Debug.Assert(result.M22 == 3); Debug.Assert(result.M23 == 0); Debug.Assert(result.M24 == 0);
                    Debug.Assert(result.M31 == 0); Debug.Assert(result.M32 == 0); Debug.Assert(result.M33 == 4); Debug.Assert(result.M34 == 0);
#if !USE_WPF
                    Debug.Assert(result.M41 == 5); Debug.Assert(result.M42 == 6); Debug.Assert(result.M43 == 7); Debug.Assert(result.M44 == 1);
#endif
                }
                // Point3D Add.
                {
                        for (int i = 0; i < 10; ++i)
                    {
                        Point3D p;
                        p = RenderToy.MathHelp.Add(new Point3D(1, 1, 1), new Point3D(i, i, i));
                        Debug.Assert(p.X == i + 1);
                        Debug.Assert(p.Y == i + 1);
                        Debug.Assert(p.Z == i + 1);
                        p = RenderToy.MathHelp.Add(new Point3D(i, i, i), new Point3D(1, 1, 1));
                        Debug.Assert(p.X == i + 1);
                        Debug.Assert(p.Y == i + 1);
                        Debug.Assert(p.Z == i + 1);
                    }
                }
                // Vector3D Length.
                {
                    for (int i = 0; i < 10; ++i)
                    {
                        Debug.Assert(RenderToy.MathHelp.Length(new Vector3D(i, 0, 0)) == i);
                        Debug.Assert(RenderToy.MathHelp.Length(new Vector3D(0, i, 0)) == i);
                        Debug.Assert(RenderToy.MathHelp.Length(new Vector3D(0, 0, i)) == i);
                    }
                    Debug.Assert(RenderToy.MathHelp.Length(new Vector3D(1, 1, 1)) == Math.Sqrt(3));
                }
#if !USE_WPF
                // WPF Equivalence.
                {
                    for (int axis = 1; axis < 10; ++axis)
                    {
                        for (int angle = 0; angle < 10; ++angle)
                        {
                            var my = new RenderToy.Quaternion(new RenderToy.Vector3D(axis, 0, 0), angle);
                            var wpf = new System.Windows.Media.Media3D.Quaternion(new System.Windows.Media.Media3D.Vector3D(axis, 0, 0), angle);
                            Debug.Assert(WithinLimits(my.X, wpf.X)); Debug.Assert(WithinLimits(my.Y, wpf.Y)); Debug.Assert(WithinLimits(my.Z, wpf.Z)); Debug.Assert(WithinLimits(my.W, wpf.W));
                            my = new RenderToy.Quaternion(new RenderToy.Vector3D(0, axis, 0), angle);
                            wpf = new System.Windows.Media.Media3D.Quaternion(new System.Windows.Media.Media3D.Vector3D(0, axis, 0), angle);
                            Debug.Assert(WithinLimits(my.X, wpf.X)); Debug.Assert(WithinLimits(my.Y, wpf.Y)); Debug.Assert(WithinLimits(my.Z, wpf.Z)); Debug.Assert(WithinLimits(my.W, wpf.W));
                            my = new RenderToy.Quaternion(new RenderToy.Vector3D(0, 0, axis), angle);
                            wpf = new System.Windows.Media.Media3D.Quaternion(new System.Windows.Media.Media3D.Vector3D(0, 0, axis), angle);
                            Debug.Assert(WithinLimits(my.X, wpf.X)); Debug.Assert(WithinLimits(my.Y, wpf.Y)); Debug.Assert(WithinLimits(my.Z, wpf.Z)); Debug.Assert(WithinLimits(my.W, wpf.W));
                        }
                    }
                }
#endif
            }
            static bool WithinLimits(double a, double b)
            {
                return Math.Abs(b - a) < 0.000000001;
            }
        }
    }
}