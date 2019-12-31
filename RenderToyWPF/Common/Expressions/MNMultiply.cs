using RenderToy.Utility;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class MNMultiply : MNBinary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "X"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Multiply(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
}