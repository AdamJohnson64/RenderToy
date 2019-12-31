using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public interface IMNNode : IMaterial
    {
        Expression CreateExpression(Expression evalcontext);
    }
    public interface IMNNode<T> : IMNNode
    {
    }
}