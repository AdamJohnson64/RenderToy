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
        static Expression<Func<double, double, double>> BrickNoise2D(ParameterExpression u, ParameterExpression v)
        {
            return (Expression<Func<double, double, double>>)Expression.Lambda(
                Expression.Condition(
                    Expression.LessThan(
                        Expression.Subtract(
                            v,
                            Expression.Call(null, typeof(System.Math).GetMethod("Floor", new Type[] { typeof(double) }), new[] { v })),
                        Expression.Constant(0.5)),
                    Perlin2D.PerlinNoise2DFn.CreateInvoke(
                            Expression.Multiply(Expression.Call(null, typeof(System.Math).GetMethod("Floor", new Type[] { typeof(double) }), new[] { u }), Expression.Constant(8.0)),
                            Expression.Multiply(Expression.Call(null, typeof(System.Math).GetMethod("Floor", new Type[] { typeof(double) }), new[] { Expression.Add(v, Expression.Constant(0.5)) }), Expression.Constant(8.0))),
                    Perlin2D.PerlinNoise2DFn.CreateInvoke(
                            Expression.Multiply(Expression.Call(null, typeof(System.Math).GetMethod("Floor", new Type[] { typeof(double) }), new[] { Expression.Add(u, Expression.Constant(0.5)) }), Expression.Constant(8.0)),
                            Expression.Multiply(Expression.Call(null, typeof(System.Math).GetMethod("Floor", new Type[] { typeof(double) }), new[] { v }), Expression.Constant(8.0)))),
                "BrickNoise2D", new ParameterExpression[] { u, v });
        }
        static Expression<Func<double, double, double>> CreateExpressionFn = ExpressionReducer.CreateLambda(BrickNoise2D);
        public Expression CreateExpression(Expression evalcontext) => CreateExpressionFn.CreateInvoke(u.CreateExpression(evalcontext), v.CreateExpression(evalcontext));
    }
}