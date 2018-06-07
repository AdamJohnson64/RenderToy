////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    sealed class BrickNoise : MNSample2D<double>, IMNNode<double>, INamed
    {
        public static double Compute(double u, double v)
        {
            if (v - Math.Floor(v) < 0.5)
            {
                return Perlin2D.PerlinNoise2D(Math.Floor(u) * 8, Math.Floor(v + 0.5) * 8);
            }
            else
            {
                return Perlin2D.PerlinNoise2D(Math.Floor(u + 0.5) * 8, Math.Floor(v) * 8);
            }
        }
        public string Name { get { return "Brick Noise"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Invoke(CreateExpressionFn, u.CreateExpression(evalcontext), v.CreateExpression(evalcontext));
        }
        static Expression _CreateExpression()
        {
            var tempu = Expression.Parameter(typeof(double), "u");
            var tempv = Expression.Parameter(typeof(double), "v");
            return Expression.Lambda(
                Expression.Condition(
                    Expression.LessThan(
                        Expression.Subtract(
                            tempv,
                            Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv })),
                        Expression.Constant(0.5)),
                    Perlin2D.PerlinNoise2D(
                            Expression.Multiply(Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempu }), Expression.Constant(8.0)),
                            Expression.Multiply(Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { Expression.Add(tempv, Expression.Constant(0.5)) }), Expression.Constant(8.0))),
                    Perlin2D.PerlinNoise2D(
                            Expression.Multiply(Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { Expression.Add(tempu, Expression.Constant(0.5)) }), Expression.Constant(8.0)),
                            Expression.Multiply(Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv }), Expression.Constant(8.0)))),
                "BrickNoise",
                new ParameterExpression[] { tempu, tempv });
        }
        static Expression CreateExpressionFn = _CreateExpression();
    }
}