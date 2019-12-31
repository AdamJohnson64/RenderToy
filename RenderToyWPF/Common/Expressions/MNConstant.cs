using RenderToy.Utility;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class MNConstant : IMNNode<double>, INamed
    {
        public string Name { get { return value.ToString(); } }
        public bool IsConstant() { return true; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Constant(value);
        }
        public double Value { get { return value; } set { this.value = value; } }
        double value;
    }
}