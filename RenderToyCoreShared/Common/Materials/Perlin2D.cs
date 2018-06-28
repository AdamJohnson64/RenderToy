////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Expressions;
using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    sealed class Perlin2D : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Perlin (2D)"; } }
        static Expression<Func<int, int, int>> Temp1Fn2 = (x, y) => x + y * 57;
        static Expression<Func<int, int, int>> Temp1Fn = ExpressionReplaceCalls.Replace(Temp1Fn2, "Random2DMix");
        static Expression<Func<int, int>> Temp2Fn2 = (n) => (n << 13) ^ n;
        static Expression<Func<int, int>> Temp2Fn = ExpressionReplaceCalls.Replace(Temp2Fn2, "RandomExtend");
        static Expression<Func<int, double>> Temp3Fn2 = (n) => 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        static Expression<Func<int, double>> Temp3Fn = ExpressionReplaceCalls.Replace(Temp3Fn2, "RandomDouble");
        static Expression Random2D(Expression x, Expression y) => Temp3Fn.CreateInvoke(Temp2Fn.CreateInvoke(Temp1Fn.CreateInvoke(x, y)));
        static Expression<Func<double, double, double>> Noise2D(ParameterExpression x, ParameterExpression y) => (Expression<Func<double, double, double>>)Expression.Lambda(Random2D(Expression.Convert(x, typeof(int)), Expression.Convert(y, typeof(int))), "Noise2D", new[] { x, y });
        static Expression<Func<double, double, double>> Noise2DFn = ExpressionReducer.CreateLambda(Noise2D); 
        static Expression<Func<double, double, double>> SmoothNoise(ParameterExpression x, ParameterExpression y)
        {
            var xminus1 = Expression.Subtract(x, Expression.Constant(1.0));
            var xplus1 = Expression.Add(x, Expression.Constant(1.0));
            var yminus1 = Expression.Subtract(y, Expression.Constant(1.0));
            var yplus1 = Expression.Add(y, Expression.Constant(1.0));
            var corners =
                Expression.Divide(
                    Expression.Add(Noise2DFn.CreateInvoke(xminus1, yminus1), Expression.Add(Noise2DFn.CreateInvoke(xplus1, yminus1), Expression.Add(Noise2DFn.CreateInvoke(xminus1, yplus1), Noise2DFn.CreateInvoke(xplus1, yplus1)))),
                    Expression.Constant(16.0));
            var sides =
                Expression.Divide(
                    Expression.Add(Noise2DFn.CreateInvoke(xminus1, y), Expression.Add(Noise2DFn.CreateInvoke(xplus1, y), Expression.Add(Noise2DFn.CreateInvoke(x, yminus1), Noise2DFn.CreateInvoke(x, yplus1)))),
                    Expression.Constant(8.0));
            var center = Expression.Divide(Noise2DFn.CreateInvoke(x, y), Expression.Constant(4.0));
            return (Expression<Func<double, double, double>>)Expression.Lambda(Expression.Add(corners, Expression.Add(sides, center)), new[] { x, y });
        }
        static Expression<Func<double, double, double>> SmoothNoiseFn = ExpressionReducer.CreateLambda(SmoothNoise);
        static Expression<Func<double, double, double>> InterpolatedNoise(ParameterExpression x, ParameterExpression y)
        {
            var tempix = Expression.Parameter(typeof(double), "SampleURounded");
            var tempiy = Expression.Parameter(typeof(double), "SampleVRounded");
            var tempfx = Expression.Parameter(typeof(double), "TiledU");
            var tempfy = Expression.Parameter(typeof(double), "TiledV");
            return (Expression<Func<double, double, double>>)Expression.Lambda(
                Expression.Block(
                    new ParameterExpression[] { tempix, tempiy, tempfx, tempfy },
                    new Expression[]
                    {
                        Expression.Assign(tempix, Expression.Convert(Expression.Convert(x, typeof(int)), typeof(double))),
                        Expression.Assign(tempiy, Expression.Convert(Expression.Convert(y, typeof(int)), typeof(double))),
                        Expression.Assign(tempfx, Expression.Subtract(x, tempix)),
                        Expression.Assign(tempfy, Expression.Subtract(y, tempiy)),
                        InvokeLerp(
                            InvokeLerp(SmoothNoiseFn.CreateInvoke(tempix, tempiy), SmoothNoiseFn.CreateInvoke(Expression.Add(tempix, Expression.Constant(1.0)), tempiy), tempfx),
                            InvokeLerp(SmoothNoiseFn.CreateInvoke(tempix, Expression.Add(tempiy, Expression.Constant(1.0))), SmoothNoiseFn.CreateInvoke(Expression.Add(tempix, Expression.Constant(1.0)), Expression.Add(tempiy, Expression.Constant(1.0))), tempfx),
                            tempfy)
                    }), "InterpolatedNoise", new[] { x, y });
        }
        static Expression<Func<double, double, double>> InterpolatedNoiseFn = ExpressionReducer.CreateLambda(InterpolatedNoise);
        static Expression<Func<double, double, double>> PerlinNoise2D(ParameterExpression x, ParameterExpression y)
        {
            return (Expression<Func<double, double, double>>)Expression.Lambda(
                Expression.Add(
                Expression.Multiply(InterpolatedNoiseFn.CreateInvoke(Expression.Multiply(x, Expression.Constant(1.0)), Expression.Multiply(y, Expression.Constant(1.0))), Expression.Constant(1.0)),
                Expression.Add(
                    Expression.Multiply(InterpolatedNoiseFn.CreateInvoke(Expression.Multiply(x, Expression.Constant(2.0)), Expression.Multiply(y, Expression.Constant(2.0))), Expression.Constant(0.5)),
                    Expression.Add(
                        Expression.Multiply(InterpolatedNoiseFn.CreateInvoke(Expression.Multiply(x, Expression.Constant(4.0)), Expression.Multiply(y, Expression.Constant(4.0))), Expression.Constant(0.25)),
                        Expression.Multiply(InterpolatedNoiseFn.CreateInvoke(Expression.Multiply(x, Expression.Constant(8.0)), Expression.Multiply(y, Expression.Constant(8.0))), Expression.Constant(0.125))))),
                "PerlinNoise2D", new[] { x, y });
        }
        public static Expression<Func<double, double, double>> PerlinNoise2DFn = ExpressionReducer.CreateLambda(PerlinNoise2D);
        public Expression CreateExpression(Expression evalcontext) => PerlinNoise2DFn.CreateInvoke(u.CreateExpression(evalcontext), v.CreateExpression(evalcontext));
    }
}