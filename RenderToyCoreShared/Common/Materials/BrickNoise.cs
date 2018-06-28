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
    sealed class BrickNoise : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Brick Noise"; } }
        static Expression<Func<double, double>> FloorFn = (a) => System.Math.Floor(a);
        static Expression<Func<double, double, double>> BrickNoiseFn = (u, v) =>
            v - Floor.Call(v) < 0.5
                ? Perlin2D.Perlin.Call(Floor.Call(u) * 8, Floor.Call(v + 0.5) * 8)
                : Perlin2D.Perlin.Call(Floor.Call(u + 0.5) * 8, Floor.Call(v) * 8);
        static readonly ExpressionFlatten<Func<double, double, double>> Noise = BrickNoiseFn.Rename("BrickNoise").Flatten();
        public Expression CreateExpression(Expression evalcontext) => Noise.Replaced.CreateInvoke(u.CreateExpression(evalcontext), v.CreateExpression(evalcontext));
    }
}