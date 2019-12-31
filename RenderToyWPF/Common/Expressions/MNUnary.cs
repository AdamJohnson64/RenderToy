namespace RenderToy.Materials
{
    public abstract class MNUnary<T> : ExpressionBase
    {
        public bool IsConstant() { return value.IsConstant(); }
        public IMNNode<T> Value { get { return value; } set { this.value = value; } }
        protected IMNNode<T> value;
    }
}