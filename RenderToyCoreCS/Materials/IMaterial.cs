////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;

namespace RenderToy.Materials
{
    /// <summary>
    /// This empty IMaterial interface is only used to type-identify suitable material types.
    /// </summary>
    public interface IMaterial
    {
    }
    public class EvalContext
    {
        public double U, V;
    }
    public interface IMNNode
    {
        bool IsConstant();
    }
    public interface IMNNode<T> : IMNNode
    {
        T Eval(EvalContext context);
    }
    abstract class MNUnary<T>
    {
        public bool IsConstant() { return value.IsConstant(); }
        public IMNNode<T> Value { get { return value; } set { this.value = value; } }
        protected IMNNode<T> value;
    }
    abstract class MNBinary<T>
    {
        public bool IsConstant() { return lhs.IsConstant() && rhs.IsConstant(); }
        public IMNNode<T> Lhs { get { return lhs; } set { lhs = value; } }
        public IMNNode<T> Rhs { get { return rhs; } set { rhs = value; } }
        protected IMNNode<T> lhs, rhs;
    }
    abstract class MNSample2D<T>
    {
        public bool IsConstant() { return u.IsConstant() && v.IsConstant(); }
        public IMNNode<T> U { get { return u; } set { u = value; } }
        public IMNNode<T> V { get { return v; } set { v = value; } }
        protected IMNNode<T> u, v;
    }
    class MNTexCoordU : IMNNode<double>
    {
        public bool IsConstant() { return false; }
        public double Eval(EvalContext context) { return context.U; }
    }
    class MNTexCoordV : IMNNode<double>
    {
        public bool IsConstant() { return false; }
        public double Eval(EvalContext context) { return context.V; }
    }
    class MNConstant : IMNNode<double>
    {
        public bool IsConstant() { return true; }
        public double Eval(EvalContext context) { return value; }
        public double Value { get { return value; } set { this.value = value; } }
        protected double value;
    }
    class MNVector4D : IMNNode<Vector4D>
    {
        public bool IsConstant() { return r.IsConstant() && g.IsConstant() && b.IsConstant() && a.IsConstant(); }
        public Vector4D Eval(EvalContext context) { return new Vector4D(r.Eval(context), g.Eval(context), b.Eval(context), a.Eval(context)); }
        public IMNNode<double> R { get { return r; } set { r = value; } }
        public IMNNode<double> G { get { return g; } set { g = value; } }
        public IMNNode<double> B { get { return b; } set { b = value; } }
        public IMNNode<double> A { get { return a; } set { a = value; } }
        protected IMNNode<double> r, g, b, a;
    }
    class MNAdd : MNBinary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return lhs.Eval(context) + rhs.Eval(context); }
    }
    class MNSubtract : MNBinary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return lhs.Eval(context) - rhs.Eval(context); }
    }
    class MNMultiply : MNBinary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return lhs.Eval(context) * rhs.Eval(context); }
    }
    class MNSaturate : MNUnary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { double v = value.Eval(context); return v < 0 ? 0 : (v < 1 ? v : 1); }
    }
    class MNThreshold : MNUnary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return value.Eval(context) < 0.5 ? 0 : 1; }
    }
    class MNLerp : IMNNode<double>
    {
        public bool IsConstant() { return value0.IsConstant() && value1.IsConstant() && factor.IsConstant(); }
        public double Eval(EvalContext context) { double f = factor.Eval(context); return value0.Eval(context) * (1 - f) + value1.Eval(context) * f; }
        public IMNNode<double> Value0 { get { return value0; } set { value0 = value; } }
        public IMNNode<double> Value1 { get { return value1; } set { value1 = value; } }
        public IMNNode<double> Factor { get { return factor; } set { factor = value; } }
        protected IMNNode<double> value0, value1, factor;
    }
    class MNBrick : MNSample2D<Vector4D>, IMNNode<Vector4D>
    {
        public static double BrickMask(double u, double v)
        {
            const double MortarWidth = 0.025;
            u = u - Math.Floor(u);
            v = v - Math.Floor(v);
            if (v < MortarWidth) return 0;
            else if (v < 0.5 - MortarWidth)
            {
                if (u < MortarWidth) return 0;
                else if (u < 1.0 - MortarWidth) return 1;
                else return 0;
            }
            else if (v < 0.5 + MortarWidth) return 0;
            else if (v < 1.0 - MortarWidth)
            {
                if (u < 0.5 - MortarWidth) return 1;
                else if (u < 0.5 + MortarWidth) return 0;
                else return 1;
            }
            else return 0;
        }
        public static double BrickNoise(double u, double v)
        {
            if (v - Math.Floor(v) < 0.5)
            {
                u = Math.Floor(u);
                v = Math.Floor(v + 0.5);
            }
            else
            {
                u = Math.Floor(u + 0.5);
                v = Math.Floor(v);
            }
            return MNPerlin2D.PerlinNoise2D(u * 8, v * 8);
        }
        public Vector4D SampleTexture(double u, double v)
        {
            u = u * 4;
            v = v * 4;
            // Calculate Perlin sources.
            double perlinlow = MNPerlin2D.PerlinNoise2D(u * 16, v * 16) * 0.1;
            double perlinmid = MNPerlin2D.PerlinNoise2D(u * 64, v * 64);
            perlinmid = MathHelp.Saturate(perlinmid) * 1.25;
            double perlinhigh = MNPerlin2D.PerlinNoise2D(u * 512, v * 512) * 0.1;
            double perlinband = MNPerlin2D.PerlinNoise2D(u * 64, v * 512) * 0.2;
            // Calculate brick and mortar colors.
            Vector4D BrickColor = new Vector4D(0.5 + perlinband + perlinlow + BrickNoise(u, v) * 0.1, 0.0, 0.0, 1);
            Vector4D MortarColor = new Vector4D(0.4 + perlinhigh + perlinlow, 0.4 + perlinhigh + perlinlow, 0.4 + perlinhigh + perlinlow, 1);
            // Calculate brickness mask.
            double brickness = BrickMask(u, v) - perlinmid;
            return brickness < 0.5 ? MortarColor : BrickColor;
        }
        public Vector4D Eval(EvalContext context)
        {
            return SampleTexture(context.U, context.V);
        }
    }
    class MNBrickMask : MNSample2D<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return MNBrick.BrickMask(u.Eval(context), v.Eval(context)); }
    }
    class MNBrickNoise : MNSample2D<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return MNBrick.BrickNoise(u.Eval(context), v.Eval(context)); }
    }
    class MNCheckerboard : IMNNode<Vector4D>, IMaterial
    {
        public Vector4D Eval(EvalContext context)
        {
            int u = (int)((context.U - Math.Floor(context.U)) * 2);
            int v = (int)((context.V - Math.Floor(context.V)) * 2);
            return ((u + v) & 1) == 0 ? Color1.Eval(context) : Color2.Eval(context);
        }
        public bool IsConstant()
        {
            return false;
        }
        public IMNNode<Vector4D> Color1 { get { return color1; } set { color1 = value; } }
        public IMNNode<Vector4D> Color2 { get { return color2; } set { color2 = value; } }
        protected IMNNode<Vector4D> color1, color2;
    }
    class MNMarbleBlack : IMNNode<Vector4D>
    {
        public Vector4D Eval(EvalContext context)
        {
            double perlinlow = MNPerlin2D.PerlinNoise2D(context.U * 5, context.V * 5);
            double perlinhigh = MNPerlin2D.PerlinNoise2D(context.U * 50, context.V * 50);
            double v1 = (1 + Math.Sin((context.U + perlinlow / 2 + perlinhigh / 5) * 50)) / 2;
            double v2 = (1 + Math.Sin((context.U + 50 + perlinlow / 2) * 100)) / 2;
            v1 = Math.Pow(v1, 20) * 0.8;
            v2 = Math.Pow(v2, 20) * 0.2;
            double v = v1 + v2;
            return new Vector4D(v, v, v, 1);
        }
        public bool IsConstant()
        {
            return false;
        }
    }
    class MNMarbleWhite : IMNNode<Vector4D>
    {
        public Vector4D Eval(EvalContext context)
        {
            double perlinlow = MNPerlin2D.PerlinNoise2D(context.U * 5, context.V * 5);
            double perlinhigh = MNPerlin2D.PerlinNoise2D(context.U * 50, context.V * 50);
            double v1 = (1 + Math.Sin((context.U + perlinlow / 2 + perlinhigh / 5) * 50)) / 2;
            double v2 = (1 + Math.Sin((context.U + 50 + perlinlow / 2) * 100)) / 2;
            v1 = Math.Pow(v1, 20) * 0.8;
            v2 = Math.Pow(v2, 20) * 0.2;
            double v = 1 - (v1 + v2);
            return new Vector4D(v, v, v, 1);
        }
        public bool IsConstant()
        {
            return false;
        }
    }
    class MNPerlin2D : MNSample2D<double>, IMNNode<double>
    {
        static double Interpolate(double a, double b, double x)
        {
            return a * (1 - x) + b * x;
        }
        static double Noise1(int x, int y)
        {
            int n = x + y * 57;
            n = (n << 13) ^ n;
            return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        }
        static double Noise(double x, double y)
        {
            return Noise1((int)x, (int)y);
        }
        static double SmoothNoise(double x, double y)
        {
            double corners = (Noise(x - 1, y - 1) + Noise(x + 1, y - 1) + Noise(x - 1, y + 1) + Noise(x + 1, y + 1)) / 16;
            double sides = (Noise(x - 1, y) + Noise(x + 1, y) + Noise(x, y - 1) + Noise(x, y + 1)) / 8;
            double center = Noise(x, y) / 4;
            return corners + sides + center;
        }
        static double InterpolatedNoise(double x, double y)
        {
            int ix = (int)x;
            double fx = x - ix;
            int iy = (int)y;
            double fy = y - iy;
            double v1 = SmoothNoise(ix, iy);
            double v2 = SmoothNoise(ix + 1, iy);
            double v3 = SmoothNoise(ix, iy + 1);
            double v4 = SmoothNoise(ix + 1, iy + 1);
            double i1 = Interpolate(v1, v2, fx);
            double i2 = Interpolate(v3, v4, fx);
            return Interpolate(i1, i2, fy);
        }
        public static double PerlinNoise2D(double x, double y)
        {
            double sum = 0;
            for (int i = 0; i < 4; ++i)
            {
                double frequency = Math.Pow(2, i);
                double amplitude = Math.Pow(0.5, i);
                sum = sum + InterpolatedNoise(x * frequency, y * frequency) * amplitude;
            }
            return sum;
        }
        public Vector4D SampleTexture(double u, double v)
        {
            double p = PerlinNoise2D(u, v);
            return new Vector4D(p, p, p, 1);
        }
        public double Eval(EvalContext context) { return PerlinNoise2D(u.Eval(context), v.Eval(context)); }
    }
}