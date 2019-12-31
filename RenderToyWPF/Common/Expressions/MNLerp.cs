using RenderToy.Expressions;
using RenderToy.Utility;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class MNLerp : ExpressionBase, IMNNode<double>, INamed
    {
        public string Name { get { return "Lerp"; } }
        public bool IsConstant() { return value0.IsConstant() && value1.IsConstant() && factor.IsConstant(); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Lerp.Replaced.CreateInvoke(value0.CreateExpression(evalcontext), value1.CreateExpression(evalcontext), factor.CreateExpression(evalcontext));
        }
        public IMNNode<double> Value0 { get { return value0; } set { value0 = value; } }
        public IMNNode<double> Value1 { get { return value1; } set { value1 = value; } }
        public IMNNode<double> Factor { get { return factor; } set { factor = value; } }
        IMNNode<double> value0, value1, factor;
    }
}