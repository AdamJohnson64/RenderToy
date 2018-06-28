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
        static Expression<Func<double, double, double>> BrickNoise2D(ParameterExpression u, ParameterExpression v)
        {
            return (Expression<Func<double, double, double>>)Expression.Lambda(
                Expression.Condition(
                    Expression.LessThan(
                        Expression.Subtract(
                            v,
                            FloorFn.CreateInvoke(v)),
                        Expression.Constant(0.5)),
                    Perlin2D.PerlinNoise2DFn.CreateInvoke(
                            Expression.Multiply(FloorFn.CreateInvoke(u), Expression.Constant(8.0)),
                            Expression.Multiply(FloorFn.CreateInvoke(Expression.Add(v, Expression.Constant(0.5))), Expression.Constant(8.0))),
                    Perlin2D.PerlinNoise2DFn.CreateInvoke(
                            Expression.Multiply(FloorFn.CreateInvoke(Expression.Add(u, Expression.Constant(0.5))), Expression.Constant(8.0)),
                            Expression.Multiply(FloorFn.CreateInvoke(v), Expression.Constant(8.0)))),
                "BrickNoise2D", new ParameterExpression[] { u, v });
        }
        static Expression<Func<double, double, double>> BrickNoise2DFn = ExpressionReducer.CreateLambda(BrickNoise2D);
        public Expression CreateExpression(Expression evalcontext) => BrickNoise2DFn.CreateInvoke(u.CreateExpression(evalcontext), v.CreateExpression(evalcontext));
    }
}