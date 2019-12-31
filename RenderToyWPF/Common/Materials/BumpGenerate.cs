using RenderToy.Math;
using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class BumpGenerate : MNSample2D<Vector4D>, IMNNode<Vector4D>, INamed
    {
        public string Name { get => "Bump Generate"; }
        public new bool IsConstant() => displacement.IsConstant();
        static Expression _ReconstructSampler2(ParameterExpression ec, ParameterExpression newu, ParameterExpression newv)
        {
            var exprec = Expression.Variable(typeof(EvalContext));
            return Expression.Block(typeof(EvalContext),
                new ParameterExpression[] { exprec },
                new Expression[]
                {
                    Expression.Assign(exprec, Expression.New(typeof(EvalContext).GetConstructor(new Type[] { typeof(EvalContext) }), ec)),
                    Expression.Assign(Expression.Field(exprec, "U"), newu),
                    Expression.Assign(Expression.Field(exprec, "V"), newv),
                    exprec
                }
            );
        }
        static LambdaExpression _ReconstructSampler()
        {
            var ec = Expression.Parameter(typeof(EvalContext), "Context");
            var du = Expression.Parameter(typeof(double), "NewU");
            var dv = Expression.Parameter(typeof(double), "NewV");
            return Expression.Lambda(_ReconstructSampler2(ec, du, dv), new ParameterExpression[] { ec, du, dv });
        }
        static LambdaExpression ReconstructSampler = _ReconstructSampler();
        public Expression CreateExpression(Expression evalcontext)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            var du1 = Expression.Parameter(typeof(double), "NegU");
            var du2 = Expression.Parameter(typeof(double), "PosU");
            var dv1 = Expression.Parameter(typeof(double), "NegV");
            var dv2 = Expression.Parameter(typeof(double), "PosV");
            var normal = Expression.Parameter(typeof(Vector3D), "Normal");
            return Expression.Block(typeof(Vector4D), new ParameterExpression[] { u, v, du1, du2, dv1, dv2, normal },
                new Expression[]
                {
                    Expression.Assign(u, U.CreateExpression(evalcontext)),
                    Expression.Assign(v, V.CreateExpression(evalcontext)),
                    Expression.Assign(du1, Displacement.CreateExpression(Expression.Invoke(ReconstructSampler, evalcontext, Expression.Subtract(u, Expression.Constant(0.001)), v))),
                    Expression.Assign(du2, Displacement.CreateExpression(Expression.Invoke(ReconstructSampler, evalcontext, Expression.Add(u, Expression.Constant(0.001)), v))),
                    Expression.Assign(dv1, Displacement.CreateExpression(Expression.Invoke(ReconstructSampler, evalcontext, u, Expression.Subtract(v, Expression.Constant(0.001))))),
                    Expression.Assign(dv2, Displacement.CreateExpression(Expression.Invoke(ReconstructSampler, evalcontext, u, Expression.Add(v, Expression.Constant(0.001))))),
                    Expression.Assign(normal, Expression.Call(null, typeof(MathHelp).GetMethod("Normalized", new Type[] { typeof(Vector3D) }),
                        Expression.New(typeof(Vector3D).GetConstructor(new Type[] { typeof(double), typeof(double), typeof(double) }),
                        Expression.Divide(Expression.Subtract(du1, du2), Expression.Constant(0.002)),
                        Expression.Divide(Expression.Subtract(dv1, dv2), Expression.Constant(0.002)),
                        Expression.Constant(1.0)
                    ))),
                    Expression.New(typeof(Vector4D).GetConstructor(new Type[] { typeof(double), typeof(double), typeof(double), typeof(double) }),
                        Expression.Add(Expression.Multiply(Expression.Field(normal, "X"), Expression.Constant(0.5)), Expression.Constant(0.5)),
                        Expression.Add(Expression.Multiply(Expression.Field(normal, "Y"), Expression.Constant(0.5)), Expression.Constant(0.5)),
                        Expression.Add(Expression.Multiply(Expression.Field(normal, "Z"), Expression.Constant(0.5)), Expression.Constant(0.5)),
                        Expression.Constant(1.0))
                });
        }
        public IMNNode<double> Displacement
        {
            get => displacement;
            set => displacement = value;
        }
        IMNNode<double> displacement;
    }
}