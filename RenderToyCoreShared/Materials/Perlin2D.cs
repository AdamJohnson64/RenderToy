////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    sealed class Perlin2D : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Perlin (2D)"; } }
        public static Expression<Func<int, int, int>> Temp1 = (x, y) => x + y * 57;
        public static Expression<Func<int, int>> Temp2 = (n) => (n << 13) ^ n;
        public static Expression<Func<int, double>> Temp3 = (n) => 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        public static double Random2D(int x, int y)
        {
            int n = x + y * 57;
            n = (n << 13) ^ n;
            return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        }
        public static Expression Random2D(Expression x, Expression y)
        {
            var _temp = Expression.Invoke(Temp1, x, y);
            var _temp2 = Expression.Invoke(Temp2, _temp);
            return Expression.Invoke(Temp3, _temp2);
        }
        public static double Noise2D(double x, double y)
        {
            return Random2D((int)x, (int)y);
        }
        public static Expression _Noise2D(Expression x, Expression y)
        {
            return Random2D(Expression.Convert(x, typeof(int)), Expression.Convert(y, typeof(int)));
        }
        public static Expression _Noise2D()
        {
            var xp = Expression.Parameter(typeof(double), "u");
            var yp = Expression.Parameter(typeof(double), "v");
            return Expression.Lambda(_Noise2D(xp, yp), "Noise2D", new[] { xp, yp });
        }
        public static Expression Noise2DFn = _Noise2D();
        public static Expression Noise2D(Expression x, Expression y)
        {
            return Expression.Invoke(Noise2DFn, x, y);
        }
        public static double SmoothNoise(double x, double y)
        {
            double corners = (Noise2D(x - 1, y - 1) + Noise2D(x + 1, y - 1) + Noise2D(x - 1, y + 1) + Noise2D(x + 1, y + 1)) / 16;
            double sides = (Noise2D(x - 1, y) + Noise2D(x + 1, y) + Noise2D(x, y - 1) + Noise2D(x, y + 1)) / 8;
            double center = Noise2D(x, y) / 4;
            return corners + sides + center;
        }
        public static Expression _SmoothNoise(Expression x, Expression y)
        {
            var xminus1 = Expression.Subtract(x, Expression.Constant(1.0));
            var xplus1 = Expression.Add(x, Expression.Constant(1.0));
            var yminus1 = Expression.Subtract(y, Expression.Constant(1.0));
            var yplus1 = Expression.Add(y, Expression.Constant(1.0));
            var corners =
                Expression.Divide(
                    Expression.Add(Noise2D(xminus1, yminus1), Expression.Add(Noise2D(xplus1, yminus1), Expression.Add(Noise2D(xminus1, yplus1), Noise2D(xplus1, yplus1)))),
                    Expression.Constant(16.0));
            var sides =
                Expression.Divide(
                    Expression.Add(Noise2D(xminus1, y), Expression.Add(Noise2D(xplus1, y), Expression.Add(Noise2D(x, yminus1), Noise2D(x, yplus1)))),
                    Expression.Constant(8.0));
            var center = Expression.Divide(Noise2D(x, y), Expression.Constant(4.0));
            return Expression.Add(corners, Expression.Add(sides, center));
        }
        public static Expression _SmoothNoise()
        {
            var xp = Expression.Parameter(typeof(double), "u");
            var yp = Expression.Parameter(typeof(double), "v");
            return Expression.Lambda(_SmoothNoise(xp, yp), "SmoothNoise", new[] { xp, yp });
        }
        public static Expression SmoothNoiseFn = _SmoothNoise();
        public static Expression SmoothNoise(Expression x, Expression y)
        {
            return Expression.Invoke(SmoothNoiseFn, x, y);
        }
        public static double InterpolatedNoise(double x, double y)
        {
            int ix = (int)x;
            double fx = x - ix;
            int iy = (int)y;
            double fy = y - iy;
            return Lerp(
                Lerp(SmoothNoise(ix, iy), SmoothNoise(ix + 1, iy), fx),
                Lerp(SmoothNoise(ix, iy + 1), SmoothNoise(ix + 1, iy + 1), fx),
                fy);
        }
        public static Expression _InterpolatedNoise(Expression x, Expression y)
        {
            var tempix = Expression.Parameter(typeof(double), "SampleURounded");
            var tempiy = Expression.Parameter(typeof(double), "SampleVRounded");
            var tempfx = Expression.Parameter(typeof(double), "TiledU");
            var tempfy = Expression.Parameter(typeof(double), "TiledV");
            return Expression.Block(
                new ParameterExpression[] { tempix, tempiy, tempfx, tempfy },
                new Expression[]
                {
                    Expression.Assign(tempix, Expression.Convert(Expression.Convert(x, typeof(int)), typeof(double))),
                    Expression.Assign(tempiy, Expression.Convert(Expression.Convert(y, typeof(int)), typeof(double))),
                    Expression.Assign(tempfx, Expression.Subtract(x, tempix)),
                    Expression.Assign(tempfy, Expression.Subtract(y, tempiy)),
                    InvokeLerp(
                        InvokeLerp(SmoothNoise(tempix, tempiy), SmoothNoise(Expression.Add(tempix, Expression.Constant(1.0)), tempiy), tempfx),
                        InvokeLerp(SmoothNoise(tempix, Expression.Add(tempiy, Expression.Constant(1.0))), SmoothNoise(Expression.Add(tempix, Expression.Constant(1.0)), Expression.Add(tempiy, Expression.Constant(1.0))), tempfx),
                        tempfy)
                });
        }
        public static Expression _InterpolatedNoise()
        {
            var x = Expression.Parameter(typeof(double), "u");
            var y = Expression.Parameter(typeof(double), "v");
            return Expression.Lambda(_InterpolatedNoise(x, y), "InterpolatedNoise", new[] { x, y });
        }
        public static Expression InterpolatedNoiseFn = _InterpolatedNoise();
        public static Expression InterpolatedNoise(Expression x, Expression y)
        {
            return Expression.Invoke(InterpolatedNoiseFn, x, y);
        }
        public static double PerlinNoise2D(double x, double y)
        {
            return
                InterpolatedNoise(x * 1.0, y * 1.0) * 1.0 +
                InterpolatedNoise(x * 2.0, y * 2.0) * 0.5 +
                InterpolatedNoise(x * 4.0, y * 4.0) * 0.25 +
                InterpolatedNoise(x * 8.0, y * 8.0) * 0.125;
        }
        public static Expression _PerlinNoise2D(Expression x, Expression y)
        {
            return Expression.Add(
                Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(1.0)), Expression.Multiply(y, Expression.Constant(1.0))), Expression.Constant(1.0)),
                Expression.Add(
                    Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(2.0)), Expression.Multiply(y, Expression.Constant(2.0))), Expression.Constant(0.5)),
                    Expression.Add(
                        Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(4.0)), Expression.Multiply(y, Expression.Constant(4.0))), Expression.Constant(0.25)),
                        Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(8.0)), Expression.Multiply(y, Expression.Constant(8.0))), Expression.Constant(0.125)))));
        }
        public static Expression _PerlinNoise2D()
        {
            var x = Expression.Parameter(typeof(double), "u");
            var y = Expression.Parameter(typeof(double), "v");
            return Expression.Lambda(_PerlinNoise2D(x, y), "PerlinNoise2D", new[] { x, y });
        }
        public static Expression PerlinNoiseFn = _PerlinNoise2D();
        public static Expression PerlinNoise2D(Expression x, Expression y)
        {
            return Expression.Invoke(PerlinNoiseFn, x, y);
        }
        public Vector4D SampleTexture(double u, double v)
        {
            double p = PerlinNoise2D(u, v);
            return new Vector4D(p, p, p, 1);
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            return PerlinNoise2D(u.CreateExpression(evalcontext), v.CreateExpression(evalcontext));
        }
    }
}