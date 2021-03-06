﻿using RenderToy.Expressions;
using System;
using System.Diagnostics;
using System.Linq.Expressions;

namespace RenderToy.Math
{
    public static partial class MathHelp
    {
        #region - Section : Basic Math Extensions -
        public static double Clamp(double v, double min, double max) { return v < min ? min : (v < max ? v : max); }
        public static double Saturate(double v) { return Clamp(v, 0, 1); }
        #endregion
        #region - Section : Simple Functions (Type-compatible with WPF) -
        public static Matrix3D CreateMatrixIdentity() { return new Matrix3D(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); }
        public static Matrix3D CreateMatrixScale(double x, double y, double z) { return new Matrix3D(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1); }
        public static Matrix3D CreateMatrixTranslate(double x, double y, double z) { return new Matrix3D(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1); }
        public static Quaternion CreateQuaternionIdentity() { return new Quaternion(0, 0, 0, 1); }
        public static Vector2D Add(Vector2D lhs, Vector2D rhs) { return new Vector2D(lhs.X + rhs.X, lhs.Y + rhs.Y); }
        public static Vector3D Add(Vector3D lhs, Vector3D rhs) { return new Vector3D(lhs.X + rhs.X, lhs.Y + rhs.Y, lhs.Z + rhs.Z); }
        public static Vector4D Add(Vector4D lhs, Vector4D rhs) { return new Vector4D(lhs.X + rhs.X, lhs.Y + rhs.Y, lhs.Z + rhs.Z, lhs.W + rhs.W); }
        public static Vector3D Cross(Vector3D lhs, Vector3D rhs) { return new Vector3D(lhs.Y * rhs.Z - lhs.Z * rhs.Y, lhs.Z * rhs.X - lhs.X * rhs.Z, lhs.X * rhs.Y - lhs.Y * rhs.X); }
        public static double Dot(Vector2D lhs, Vector2D rhs) { return lhs.X * rhs.X + lhs.Y * rhs.Y; }
        public static double Dot(Vector3D lhs, Vector3D rhs) { return lhs.X * rhs.X + lhs.Y * rhs.Y + lhs.Z * rhs.Z; }
        public static double Dot(Vector4D lhs, Vector4D rhs) { return lhs.X * rhs.X + lhs.Y * rhs.Y + lhs.Z * rhs.Z + lhs.W * rhs.W; }
        public static double Length(Vector2D val) { return System.Math.Sqrt(Dot(val, val)); }
        public static double Length(Vector3D val) { return System.Math.Sqrt(Dot(val, val)); }
        public static double Length(Vector4D val) { return System.Math.Sqrt(Dot(val, val)); }
        public static Vector2D Multiply(Vector2D lhs, double rhs) { return new Vector2D(lhs.X * rhs, lhs.Y * rhs); }
        public static Vector2D Multiply(double lhs, Vector2D rhs) { return Multiply(rhs, lhs); }
        public static Vector3D Multiply(Vector3D lhs, double rhs) { return new Vector3D(lhs.X * rhs, lhs.Y * rhs, lhs.Z * rhs); }
        public static Vector3D Multiply(double lhs, Vector3D rhs) { return Multiply(rhs, lhs); }
        public static Vector4D Multiply(Vector4D lhs, double rhs) { return new Vector4D(lhs.X * rhs, lhs.Y * rhs, lhs.Z * rhs, lhs.W * rhs); }
        public static Expression<Func<Vector4D, double, Vector4D>> Multiply_Vector4D_DoubleFn = (lhs, rhs) => new Vector4D(lhs.X * rhs, lhs.Y * rhs, lhs.Z * rhs, lhs.W * rhs);
        public static ExpressionFlatten<Func<Vector4D, double, Vector4D>> Multiply_Vector4D_Double = Multiply_Vector4D_DoubleFn.Rename("Multiply").Flatten();
        public static Expression<Func<double, Vector4D, Vector4D>> Multiply_Double_Vector4DFn = (rhs, lhs) => new Vector4D(lhs.X * rhs, lhs.Y * rhs, lhs.Z * rhs, lhs.W * rhs);
        public static ExpressionFlatten<Func<double, Vector4D, Vector4D>> Multiply_Double_Vector4D = Multiply_Double_Vector4DFn.Rename("Multiply").Flatten();
        public static Vector4D Multiply(double lhs, Vector4D rhs) { return Multiply(rhs, lhs); }
        public static Vector2D Negate(Vector2D val) { return new Vector2D(-val.X, -val.Y); }
        public static Vector3D Negate(Vector3D val) { return new Vector3D(-val.X, -val.Y, -val.Z); }
        public static Vector4D Negate(Vector4D val) { return new Vector4D(-val.X, -val.Y, -val.Z, -val.W); }
        public static Vector2D Normalized(Vector2D val) { return val * (1 / Length(val)); }
        public static Vector3D Normalized(Vector3D val) { return val * (1 / Length(val)); }
        public static Vector4D Normalized(Vector4D val) { return Multiply(val, 1 / Length(val)); }
        public static Vector2D Subtract(Vector2D lhs, Vector2D rhs) { return new Vector2D(lhs.X - rhs.X, lhs.Y - rhs.Y); }
        public static Vector3D Subtract(Vector3D lhs, Vector3D rhs) { return new Vector3D(lhs.X - rhs.X, lhs.Y - rhs.Y, lhs.Z - rhs.Z); }
        public static Vector4D Subtract(Vector4D lhs, Vector4D rhs) { return new Vector4D(lhs.X - rhs.X, lhs.Y - rhs.Y, lhs.Z - rhs.Z, lhs.W - rhs.W); }
        #endregion
        #region - Section : Non-Trivial Functions (Type-compatible with WPF) -
        public static Matrix3D CreateMatrixLookAt(Vector3D eye, Vector3D at, Vector3D up)
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
        public static Matrix3D CreateMatrixRotation(Quaternion q)
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
        public static Quaternion CreateQuaternionRotation(Vector3D axis, double angle)
        {
            double radians_over_2 = (angle * System.Math.PI / 180) / 2;
            double sin = System.Math.Sin(radians_over_2);
            Vector3D normalized_axis = MathHelp.Normalized(axis);
            return new Quaternion(normalized_axis.X * sin, normalized_axis.Y * sin, normalized_axis.Z * sin, System.Math.Cos(radians_over_2));
        }
        public static Quaternion CreateQuaternionRotation(Matrix3D orthonormal)
        {
            double w = System.Math.Sqrt(1 + orthonormal.M11 + orthonormal.M22 + orthonormal.M33) / 2;
            double x = (orthonormal.M23 - orthonormal.M32) / (4 * w);
            double y = (orthonormal.M31 - orthonormal.M13) / (4 * w);
            double z = (orthonormal.M12 - orthonormal.M21) / (4 * w);
            return new Quaternion(x, y, z, w);
        }
        public static Quaternion Multiply(Quaternion a, Quaternion b)
        {
            double t0 = b.W * a.W - b.X * a.X - b.Y * a.Y - b.Z * a.Z;
            double t1 = b.W * a.X + b.X * a.W - b.Y * a.Z + b.Z * a.Y;
            double t2 = b.W * a.Y + b.X * a.Z + b.Y * a.W - b.Z * a.X;
            double t3 = b.W * a.Z - b.X * a.Y + b.Y * a.X + b.Z * a.W;
            double ln = 1 / System.Math.Sqrt(t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3);
            t0 *= ln; t1 *= ln; t2 *= ln; t3 *= ln;
            return new Quaternion(t1, t2, t3, t0);
        }
        #endregion
    }
    public static partial class MathHelp
    {
        public static double Determinant(Matrix2D val)
        {
            return val.M11 * val.M22 - val.M12 * val.M21;
        }
        public static double Determinant(Matrix3D val)
        {
            return (-val.M14 * (+val.M23 * (val.M31 * val.M42 - val.M32 * val.M41) - val.M33 * (val.M21 * val.M42 - val.M22 * val.M41) + val.M43 * (val.M21 * val.M32 - val.M22 * val.M31)) + val.M24 * (+val.M13 * (val.M31 * val.M42 - val.M32 * val.M41) - val.M33 * (val.M11 * val.M42 - val.M12 * val.M41) + val.M43 * (val.M11 * val.M32 - val.M12 * val.M31)) - val.M34 * (+val.M13 * (val.M21 * val.M42 - val.M22 * val.M41) - val.M23 * (val.M11 * val.M42 - val.M12 * val.M41) + val.M43 * (val.M11 * val.M22 - val.M12 * val.M21)) + val.M44 * (+val.M13 * (val.M21 * val.M32 - val.M22 * val.M31) - val.M23 * (val.M11 * val.M32 - val.M12 * val.M31) + val.M33 * (val.M11 * val.M22 - val.M12 * val.M21)));
        }
        public static Matrix2D Invert(Matrix2D val)
        {
            double invdet = 1 / Determinant(val);
            return new Matrix2D(invdet * val.M22, invdet * -val.M12, invdet * -val.M21, invdet * val.M11);
        }
        public static Matrix3D Invert(Matrix3D val)
        {
            double invdet = 1 / Determinant(val);
            return new Matrix3D(
                invdet * (+(+val.M42 * (val.M23 * val.M34 - val.M33 * val.M24) - val.M43 * (val.M22 * val.M34 - val.M32 * val.M24) + val.M44 * (val.M22 * val.M33 - val.M32 * val.M23))),
                invdet * (-(+val.M42 * (val.M13 * val.M34 - val.M33 * val.M14) - val.M43 * (val.M12 * val.M34 - val.M32 * val.M14) + val.M44 * (val.M12 * val.M33 - val.M32 * val.M13))),
                invdet * (+(+val.M42 * (val.M13 * val.M24 - val.M23 * val.M14) - val.M43 * (val.M12 * val.M24 - val.M22 * val.M14) + val.M44 * (val.M12 * val.M23 - val.M22 * val.M13))),
                invdet * (-(+val.M32 * (val.M13 * val.M24 - val.M23 * val.M14) - val.M33 * (val.M12 * val.M24 - val.M22 * val.M14) + val.M34 * (val.M12 * val.M23 - val.M22 * val.M13))),
                invdet * (-(+val.M41 * (val.M23 * val.M34 - val.M33 * val.M24) - val.M43 * (val.M21 * val.M34 - val.M31 * val.M24) + val.M44 * (val.M21 * val.M33 - val.M31 * val.M23))),
                invdet * (+(+val.M41 * (val.M13 * val.M34 - val.M33 * val.M14) - val.M43 * (val.M11 * val.M34 - val.M31 * val.M14) + val.M44 * (val.M11 * val.M33 - val.M31 * val.M13))),
                invdet * (-(+val.M41 * (val.M13 * val.M24 - val.M23 * val.M14) - val.M43 * (val.M11 * val.M24 - val.M21 * val.M14) + val.M44 * (val.M11 * val.M23 - val.M21 * val.M13))),
                invdet * (+(+val.M31 * (val.M13 * val.M24 - val.M23 * val.M14) - val.M33 * (val.M11 * val.M24 - val.M21 * val.M14) + val.M34 * (val.M11 * val.M23 - val.M21 * val.M13))),
                invdet * (+(+val.M41 * (val.M22 * val.M34 - val.M32 * val.M24) - val.M42 * (val.M21 * val.M34 - val.M31 * val.M24) + val.M44 * (val.M21 * val.M32 - val.M31 * val.M22))),
                invdet * (-(+val.M41 * (val.M12 * val.M34 - val.M32 * val.M14) - val.M42 * (val.M11 * val.M34 - val.M31 * val.M14) + val.M44 * (val.M11 * val.M32 - val.M31 * val.M12))),
                invdet * (+(+val.M41 * (val.M12 * val.M24 - val.M22 * val.M14) - val.M42 * (val.M11 * val.M24 - val.M21 * val.M14) + val.M44 * (val.M11 * val.M22 - val.M21 * val.M12))),
                invdet * (-(+val.M31 * (val.M12 * val.M24 - val.M22 * val.M14) - val.M32 * (val.M11 * val.M24 - val.M21 * val.M14) + val.M34 * (val.M11 * val.M22 - val.M21 * val.M12))),
                invdet * (-(+val.M41 * (val.M22 * val.M33 - val.M32 * val.M23) - val.M42 * (val.M21 * val.M33 - val.M31 * val.M23) + val.M43 * (val.M21 * val.M32 - val.M31 * val.M22))),
                invdet * (+(+val.M41 * (val.M12 * val.M33 - val.M32 * val.M13) - val.M42 * (val.M11 * val.M33 - val.M31 * val.M13) + val.M43 * (val.M11 * val.M32 - val.M31 * val.M12))),
                invdet * (-(+val.M41 * (val.M12 * val.M23 - val.M22 * val.M13) - val.M42 * (val.M11 * val.M23 - val.M21 * val.M13) + val.M43 * (val.M11 * val.M22 - val.M21 * val.M12))),
                invdet * (+(+val.M31 * (val.M12 * val.M23 - val.M22 * val.M13) - val.M32 * (val.M11 * val.M23 - val.M21 * val.M13) + val.M33 * (val.M11 * val.M22 - val.M21 * val.M12))));
        }
        public static Matrix2D Multiply(Matrix2D a, double b)
        {
            return new Matrix2D(
                a.M11 * b, a.M12 * b,
                a.M21 * b, a.M22 * b);
        }
        public static Matrix2D Multiply(double a, Matrix2D b) { return Multiply(b, a); }
        public static Matrix3D Multiply(Matrix3D a, Matrix3D b)
        {
            return new Matrix3D(
                a.M11 * b.M11 + a.M12 * b.M21 + a.M13 * b.M31 + a.M14 * b.M41,
                a.M11 * b.M12 + a.M12 * b.M22 + a.M13 * b.M32 + a.M14 * b.M42,
                a.M11 * b.M13 + a.M12 * b.M23 + a.M13 * b.M33 + a.M14 * b.M43,
                a.M11 * b.M14 + a.M12 * b.M24 + a.M13 * b.M34 + a.M14 * b.M44,
                a.M21 * b.M11 + a.M22 * b.M21 + a.M23 * b.M31 + a.M24 * b.M41,
                a.M21 * b.M12 + a.M22 * b.M22 + a.M23 * b.M32 + a.M24 * b.M42,
                a.M21 * b.M13 + a.M22 * b.M23 + a.M23 * b.M33 + a.M24 * b.M43,
                a.M21 * b.M14 + a.M22 * b.M24 + a.M23 * b.M34 + a.M24 * b.M44,
                a.M31 * b.M11 + a.M32 * b.M21 + a.M33 * b.M31 + a.M34 * b.M41,
                a.M31 * b.M12 + a.M32 * b.M22 + a.M33 * b.M32 + a.M34 * b.M42,
                a.M31 * b.M13 + a.M32 * b.M23 + a.M33 * b.M33 + a.M34 * b.M43,
                a.M31 * b.M14 + a.M32 * b.M24 + a.M33 * b.M34 + a.M34 * b.M44,
                a.M41 * b.M11 + a.M42 * b.M21 + a.M43 * b.M31 + a.M44 * b.M41,
                a.M41 * b.M12 + a.M42 * b.M22 + a.M43 * b.M32 + a.M44 * b.M42,
                a.M41 * b.M13 + a.M42 * b.M23 + a.M43 * b.M33 + a.M44 * b.M43,
                a.M41 * b.M14 + a.M42 * b.M24 + a.M43 * b.M34 + a.M44 * b.M44);
            /*
            // This code was used to generate the code above.
            string buildcode = "return new Matrix3D(\n";
            for (int row = 0; row < 4; ++row)
            {
                for (int col = 0; col < 4; ++col)
                {
                    buildcode += "\t";
                    for (int cmp = 0; cmp < 4; ++cmp)
                    {
                        if (cmp > 0) buildcode += " + ";
                        buildcode += "a.M" + (row + 1) + (cmp + 1) + " * b.M" + (cmp + 1) + (col + 1);
                    }
                    buildcode += (row == 3 && col == 3) ? ");" : ",\n";
                }
            }
            */
        }
        public static Matrix3D Multiply(Matrix3D a, double b)
        {
            return new Matrix3D(
                a.M11 * b, a.M12 * b, a.M13 * b, a.M14 * b,
                a.M21 * b, a.M22 * b, a.M23 * b, a.M24 * b,
                a.M31 * b, a.M32 * b, a.M33 * b, a.M34 * b,
                a.M41 * b, a.M42 * b, a.M43 * b, a.M44 * b);
        }
        public static Matrix3D Multiply(double a, Matrix3D b) { return Multiply(b, a); }
        public static Vector2D Transform(Matrix2D a, Vector2D b)
        {
            return new Vector2D(a.M11 * b.X + a.M12 * b.Y, a.M21 * b.X + a.M22 * b.Y);
        }
        public static Vector4D Transform(Matrix3D a, Vector4D b)
        {
            return new Vector4D(
                a.M11 * b.X + a.M21 * b.Y + a.M31 * b.Z + a.M41 * b.W,
                a.M12 * b.X + a.M22 * b.Y + a.M32 * b.Z + a.M42 * b.W,
                a.M13 * b.X + a.M23 * b.Y + a.M33 * b.Z + a.M43 * b.W,
                a.M14 * b.X + a.M24 * b.Y + a.M34 * b.Z + a.M44 * b.W);
        }
        public static Expression<Func<Matrix3D, Vector3D, Vector3D>> TransformPoint_Matrix3D_Vector3DFn = (a, b) =>
            new Vector3D(
                    a.M11 * b.X + a.M21 * b.Y + a.M31 * b.Z + a.M41,
                    a.M12 * b.X + a.M22 * b.Y + a.M32 * b.Z + a.M42,
                    a.M13 * b.X + a.M23 * b.Y + a.M33 * b.Z + a.M43);
        public static ExpressionFlatten<Func<Matrix3D, Vector3D, Vector3D>> TransformPoint_Matrix3D_Vector3D = TransformPoint_Matrix3D_Vector3DFn.Rename("TransformPoint").Flatten();
        public static Vector3D TransformPoint(Matrix3D a, Vector3D b)
        {
            return new Vector3D(
                a.M11 * b.X + a.M21 * b.Y + a.M31 * b.Z + a.M41,
                a.M12 * b.X + a.M22 * b.Y + a.M32 * b.Z + a.M42,
                a.M13 * b.X + a.M23 * b.Y + a.M33 * b.Z + a.M43);
        }
        public static Vector3D TransformVector(Matrix3D a, Vector3D b)
        {
            return new Vector3D(
                a.M11 * b.X + a.M21 * b.Y + a.M31 * b.Z,
                a.M12 * b.X + a.M22 * b.Y + a.M32 * b.Z,
                a.M13 * b.X + a.M23 * b.Y + a.M33 * b.Z);
        }
    }
    [DebuggerDisplay("[{X}, {Y}]")]
    public struct Vector2D
    {
        public double X, Y;
        public Vector2D(double x, double y) { X = x; Y = y; }
        public static Vector2D operator -(Vector2D a) { return MathHelp.Negate(a); }
        public static Vector2D operator -(Vector2D a, Vector2D b) { return MathHelp.Subtract(a, b); }
        public static Vector2D operator +(Vector2D a, Vector2D b) { return MathHelp.Add(a, b); }
        public static Vector2D operator *(Vector2D a, double b) { return MathHelp.Multiply(a, b); }
        public static Vector2D operator *(double a, Vector2D b) { return MathHelp.Multiply(a, b); }
    }
    [DebuggerDisplay("[{X}, {Y}]")]
    public struct Vector2F
    {
        public float X, Y;
        public Vector2F(float x, float y) { X = x; Y = y; }
    }
    [DebuggerDisplay("[{X}, {Y}, {Z}]")]
    public struct Vector3D
    {
        public double X, Y, Z;
        public Vector3D(double x, double y, double z) { X = x; Y = y; Z = z; }
        public static Vector3D operator -(Vector3D a) { return MathHelp.Negate(a); }
        public static Vector3D operator -(Vector3D a, Vector3D b) { return MathHelp.Subtract(a, b); }
        public static Vector3D operator +(Vector3D a, Vector3D b) { return MathHelp.Add(a, b); }
        public static Vector3D operator *(Vector3D a, double b) { return MathHelp.Multiply(a, b); }
        public static Vector3D operator *(double a, Vector3D b) { return MathHelp.Multiply(a, b); }
    }
    [DebuggerDisplay("[{X}, {Y}, {Z}]")]
    public struct Vector3F
    {
        public float X, Y, Z;
        public Vector3F(float x, float y, float z) { X = x; Y = y; Z = z; }
    }
    [DebuggerDisplay("[{X}, {Y}, {Z}, {W}]")]
    public struct Vector4D
    {
        public double X, Y, Z, W;
        public Vector4D(double x, double y, double z, double w) { X = x;  Y = y;  Z = z; W = w; }
        public static Vector4D operator -(Vector4D a) { return MathHelp.Negate(a); }
        public static Vector4D operator +(Vector4D a, Vector4D b) { return MathHelp.Add(a, b); }
        public static Vector4D operator -(Vector4D a, Vector4D b) { return MathHelp.Subtract(a, b); }
        public static Vector4D operator *(Vector4D a, double b) { return MathHelp.Multiply(a, b); }
        public static Vector4D operator *(double a, Vector4D b) { return MathHelp.Multiply(a, b); }
    }
    [DebuggerDisplay("[{X}, {Y}, {Z}, {W}]")]
    public struct Vector4F
    {
        public float X, Y, Z, W;
        public Vector4F(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }
    }
    public struct Matrix2D
    {
        public double M11, M12;
        public double M21, M22;
        public Matrix2D(double m11, double m12, double m21, double m22) { M11 = m11; M12 = m12; M21 = m21; M22 = m22; }
        public static Matrix2D operator *(Matrix2D a, double b) { return MathHelp.Multiply(a, b); }
        public static Matrix2D operator *(double a, Matrix2D b) { return MathHelp.Multiply(a, b); }
        public Matrix2D Transform(Matrix2D a, double b) { return new Matrix2D(a.M11 * b, a.M12 * b, a.M21 * b, a.M22 * b); }
    };
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
        public static readonly Matrix3D Identity = MathHelp.CreateMatrixIdentity();
        public static Matrix3D operator*(Matrix3D a, Matrix3D b) { return MathHelp.Multiply(a, b); }
        public Vector4D Transform(Vector4D p) { return MathHelp.Transform(this, p); }
        public static bool operator==(Matrix3D a, Matrix3D b)
        {
            return
                a.M11 == b.M11 && a.M12 == b.M12 && a.M13 == b.M13 && a.M14 == b.M14 &&
                a.M21 == b.M21 && a.M22 == b.M22 && a.M23 == b.M23 && a.M24 == b.M24 &&
                a.M31 == b.M31 && a.M32 == b.M32 && a.M33 == b.M33 && a.M34 == b.M34 &&
                a.M41 == b.M41 && a.M42 == b.M42 && a.M43 == b.M43 && a.M44 == b.M44;
        }
        public static bool operator!=(Matrix3D a, Matrix3D b)
        {
            return !(a == b);
        }
        public override bool Equals(object obj)
        {
            return obj is Matrix3D other ? this == other : false;
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
    public struct Quaternion
    {
        public double X, Y, Z, W;
        public Quaternion(double x, double y, double z, double w) { X = x; Y = y; Z = z; W = w; }
        public Quaternion(Vector3D axis, double angle) { this = MathHelp.CreateQuaternionRotation(axis, angle); }
        public static Quaternion operator *(Quaternion q, Quaternion r) { return MathHelp.Multiply(q, r); }
    }
}