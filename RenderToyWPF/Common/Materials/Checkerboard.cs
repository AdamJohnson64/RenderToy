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
        static Expression<Func<double, int>> CheckerboardFn = (d) => (int)(2 * (d - Floor.Call(d)));
        static ExpressionFlatten<Func<double, int>> CheckerboardFoo = CheckerboardFn.Rename("ContinuousSquareWave").Flatten();
        static Expression<Func<int, int, bool>> TestFn = (u, v) => ((u + v) & 1) == 0;
        static ExpressionFlatten<Func<int, int, bool>> Test = TestFn.Rename("CheckerboardTest").Flatten();
        public Expression CreateExpression(Expression evalcontext)
        {
            var paramu = Expression.Parameter(typeof(int), "u");
            var paramv = Expression.Parameter(typeof(int), "v");
            var lambda = Expression.Lambda(
                Expression.Condition(
                    Test.Replaced.CreateInvoke(paramu, paramv),
                    color1.CreateExpression(evalcontext),
                    color2.CreateExpression(evalcontext)), "Checkerboard", new[] { paramu, paramv });
            return Expression.Invoke(lambda,
                CheckerboardFoo.Replaced.CreateInvoke(u.CreateExpression(evalcontext)),
                CheckerboardFoo.Replaced.CreateInvoke(v.CreateExpression(evalcontext)));
        }
        public IMNNode<Vector4D> Color1 { get { return color1; } set { color1 = value; } }
        public IMNNode<Vector4D> Color2 { get { return color2; } set { color2 = value; } }
        IMNNode<Vector4D> color1, color2;
    }
}