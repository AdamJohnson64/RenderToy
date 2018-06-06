using RenderToy.Utility;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    sealed class Checkerboard : MNSample2D<Vector4D>, IMNNode<Vector4D>, INamed
    {
        public string Name { get { return "Checkerboard"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double), "SampleU");
            var tempv = Expression.Parameter(typeof(double), "SampleV");
            var intu = Expression.Parameter(typeof(int), "TiledU");
            var intv = Expression.Parameter(typeof(int), "TiledV");
            return Expression.Block(
                typeof(Vector4D),
                new ParameterExpression[] { tempu, tempv, intu, intv },
                new Expression[]
                {
                    Expression.Assign(tempu, u.CreateExpression(evalcontext)),
                    Expression.Assign(tempv, v.CreateExpression(evalcontext)),
                    Expression.Assign(intu, Expression.Convert(Expression.Multiply(Expression.Subtract(tempu, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempu })), Expression.Constant(2.0)), typeof(int))),
                    Expression.Assign(intv, Expression.Convert(Expression.Multiply(Expression.Subtract(tempv, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv })), Expression.Constant(2.0)), typeof(int))),
                    Expression.Condition(
                        Expression.Equal(Expression.And(Expression.Add(intu, intv), Expression.Constant(1)), Expression.Constant(0)),
                        color1.CreateExpression(evalcontext),
                        color2.CreateExpression(evalcontext)),
                });
        }
        public IMNNode<Vector4D> Color1 { get { return color1; } set { color1 = value; } }
        public IMNNode<Vector4D> Color2 { get { return color2; } set { color2 = value; } }
        protected IMNNode<Vector4D> color1, color2;
    }
}