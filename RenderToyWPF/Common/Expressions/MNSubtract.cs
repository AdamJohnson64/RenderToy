using RenderToy.Utility;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class MNSubtract : MNBinary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "-"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Subtract(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
}