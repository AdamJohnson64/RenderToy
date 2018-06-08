using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    sealed class Spike : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Spike"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Variable(typeof(double));
            var tempv = Expression.Variable(typeof(double));
            return Expression.Block(
                new ParameterExpression[] { tempu, tempv },
                new Expression[]
                {
                    Expression.Assign(tempu, Expression.Subtract(u.CreateExpression(evalcontext), Expression.Constant(0.5))),
                    Expression.Assign(tempv, Expression.Subtract(v.CreateExpression(evalcontext), Expression.Constant(0.5))),
                    InvokeLerp(
                        Expression.Constant(1.0),
                        Expression.Constant(0.0),
                        MNSaturate.CreateSaturate(
                            Expression.Multiply(
                                Expression.Call(null, typeof(Math).GetMethod("Sqrt"), new Expression[]
                                {
                                    Expression.Add(Expression.Multiply(tempu, tempu), Expression.Multiply(tempv, tempv))
                                }),
                                Expression.Constant(2.0))))
                });
        }
    }
}