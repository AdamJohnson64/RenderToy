////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Expressions;
using RenderToy.Math;
using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    sealed class Checkerboard : MNSample2D<Vector4D>, IMNNode<Vector4D>, INamed
    {
        public string Name { get { return "Checkerboard"; } }
        static Expression<Func<double, double>> FloorFn2 = (d) => System.Math.Floor(d);
        static Expression<Func<double, double>> FloorFn = ExpressionReducer.Rename(FloorFn2, "Floor");
        static Expression<Func<double, double>> CheckerboardFn2()
        {
            var parameter = Expression.Parameter(typeof(double), "d");
            return (Expression<Func<double, double>>)Expression.Lambda(Expression.Multiply(Expression.Subtract(parameter, Expression.Invoke(FloorFn, parameter)), Expression.Constant(2.0)), "ContinuousSawtoothWave", new[] { parameter });
        }
        static Expression<Func<double, double>> CheckerboardFn = CheckerboardFn2();
        static Expression<Func<double, int>> SquareFn2()
        {
            var parameter = Expression.Parameter(typeof(double), "d");
            return (Expression<Func<double, int>>)Expression.Lambda(Expression.Convert(Expression.Invoke(CheckerboardFn, parameter), typeof(int)), "ContinuousSquareWave", new[] { parameter });
        }
        static Expression<Func<double, int>> SquareFn = SquareFn2();
        public Expression CreateExpression(Expression evalcontext)
        {
            var paramu = Expression.Parameter(typeof(int), "u");
            var paramv = Expression.Parameter(typeof(int), "v");
            var lambda = Expression.Lambda(
                Expression.Condition(
                    Expression.Equal(Expression.And(Expression.Add(paramu, paramv), Expression.Constant(1)), Expression.Constant(0)),
                    color1.CreateExpression(evalcontext),
                    color2.CreateExpression(evalcontext)), "Checkerboard", new[] { paramu, paramv });
            return Expression.Invoke(lambda,
                Expression.Invoke(SquareFn, u.CreateExpression(evalcontext)),
                Expression.Invoke(SquareFn, v.CreateExpression(evalcontext)));
        }
        public IMNNode<Vector4D> Color1 { get { return color1; } set { color1 = value; } }
        public IMNNode<Vector4D> Color2 { get { return color2; } set { color2 = value; } }
        IMNNode<Vector4D> color1, color2;
    }
}