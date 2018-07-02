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
    public sealed class Perlin2D : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Perlin (2D)"; } }
        static readonly Expression<Func<int, int, int>> Temp1Fn2 = (x, y) => x + y * 57;
        static readonly Expression<Func<int, int, int>> Temp1Fn = Temp1Fn2.ReplaceCalls().Rename("Random2DMix");
        static readonly Expression<Func<int, int>> Temp2Fn2 = (n) => (n << 13) ^ n;
        static readonly Expression<Func<int, int>> Temp2Fn = Temp2Fn2.ReplaceCalls().Rename("RandomExtend");
        static readonly Expression<Func<int, double>> Temp3Fn2 = (n) => 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        static readonly Expression<Func<int, double>> Temp3Fn = Temp3Fn2.ReplaceCalls().Rename("RandomDouble");
        static Expression Random2D(Expression x, Expression y) => Temp3Fn.CreateInvoke(Temp2Fn.CreateInvoke(Temp1Fn.CreateInvoke(x, y)));
        static Expression<Func<double, double, double>> Noise2D(ParameterExpression x, ParameterExpression y) => (Expression<Func<double, double, double>>)Expression.Lambda(Random2D(Expression.Convert(x, typeof(int)), Expression.Convert(y, typeof(int))), "Noise2D", new[] { x, y });
        static readonly Expression<Func<double, double, double>> Noise2DFn = ExpressionExtensions.CreateLambda(Noise2D);
        static readonly ExpressionFlatten<Func<double, double, double>> Noise2DFlat = Noise2DFn.Rename("Noise2D").Flatten();
        static readonly Expression<Func<double, double, double>> SmoothNoiseFn = (x, y) =>
            (Noise2DFlat.Call(x - 1, y - 1) + Noise2DFlat.Call(x + 1, y - 1) + Noise2DFlat.Call(x - 1, y + 1) + Noise2DFlat.Call(x + 1, y + 1)) / 16 +
            (Noise2DFlat.Call(x - 1, y) + Noise2DFlat.Call(x + 1, y) + Noise2DFlat.Call(x, y - 1) + Noise2DFlat.Call(x, y + 1)) / 8 +
            (Noise2DFlat.Call(x, y)) / 4;
        static readonly ExpressionFlatten<Func<double, double, double>> SmoothNoiseFlat = SmoothNoiseFn.Rename("SmoothNoise").Flatten();
        static readonly Expression<Func<double, double, double>> InterpolatedNoiseFn = (x,y) =>
            Lerp.Call(
                Lerp.Call(SmoothNoiseFlat.Call((int)x, (int)y), SmoothNoiseFlat.Call((int)x + 1, (int)y), x - (int)x),
                Lerp.Call(SmoothNoiseFlat.Call((int)x, (int)y + 1), SmoothNoiseFlat.Call((int)x + 1, (int)y + 1), x - (int)x),
                y - (int)y);
        static readonly ExpressionFlatten<Func<double, double, double>> InterpolatedNoiseFlat = InterpolatedNoiseFn.Rename("InterpolateNoise").Flatten();
        static readonly Expression<Func<double, double, double>> PerlinNoiseFn = (x, y) => InterpolatedNoiseFlat.Call(x * 1, y * 1) * 1 +
            InterpolatedNoiseFlat.Call(x * 2, y * 2) * 0.5 +
            InterpolatedNoiseFlat.Call(x * 4, y * 4) * 0.25 +
            InterpolatedNoiseFlat.Call(x * 8, y * 8) * 0.125;
        public static readonly ExpressionFlatten<Func<double, double, double>> Perlin = PerlinNoiseFn.Rename("PerlinNoise").Flatten();
        public Expression CreateExpression(Expression evalcontext) => Perlin.Replaced.CreateInvoke(u.CreateExpression(evalcontext), v.CreateExpression(evalcontext));
    }
}