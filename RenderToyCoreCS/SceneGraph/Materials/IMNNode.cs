using RenderToy.Textures;

namespace RenderToy.MaterialNetwork
{
    class EvalContext
    {
        public double U, V;
    }
    interface IMNNode
    {
        bool IsConstant();
    }
    interface IMNNode<T> : IMNNode
    {
        T Eval(EvalContext context);
    }
    abstract class IMUnary<T>
    {
        public bool IsConstant() { return value.IsConstant(); }
        public IMNNode<T> Value { get { return value; } set { this.value = value; } }
        protected IMNNode<T> value;
    }
    abstract class IMBinary<T>
    {
        public bool IsConstant() { return lhs.IsConstant() && rhs.IsConstant(); }
        public IMNNode<T> Lhs { get { return lhs; } set { lhs = value; } }
        public IMNNode<T> Rhs { get { return rhs; } set { rhs = value; } }
        protected IMNNode<T> lhs, rhs;
    }
    abstract class IMSample2D<T>
    {
        public bool IsConstant() { return u.IsConstant() && v.IsConstant(); }
        public IMNNode<T> U { get { return u; } set { u = value; } }
        public IMNNode<T> V { get { return v; } set { v = value; } }
        protected IMNNode<T> u, v;
    }
    class IMTexCoordU : IMNNode<double>
    {
        public bool IsConstant() { return false; }
        public double Eval(EvalContext context) { return context.U; }
    }
    class IMTexCoordV : IMNNode<double>
    {
        public bool IsConstant() { return false; }
        public double Eval(EvalContext context) { return context.V; }
    }
    class IMConstant<T> : IMNNode<T>
    {
        public bool IsConstant() { return true; }
        public T Eval(EvalContext context) { return value; }
        public T Value { get { return value; } set { this.value = value; } }
        protected T value;
    }
    class IMVector4D : IMNNode<Vector4D>
    {
        public bool IsConstant() { return r.IsConstant() && g.IsConstant() && b.IsConstant() && a.IsConstant(); }
        public Vector4D Eval(EvalContext context) { return new Vector4D(r.Eval(context), g.Eval(context), b.Eval(context), a.Eval(context)); }
        public IMNNode<double> R { get { return r; } set { r = value; } }
        public IMNNode<double> G { get { return g; } set { g = value; } }
        public IMNNode<double> B { get { return b; } set { b = value; } }
        public IMNNode<double> A { get { return a; } set { a = value; } }
        protected IMNNode<double> r, g, b, a;
    }
    class IMAdd : IMBinary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return lhs.Eval(context) + rhs.Eval(context); }
    }
    class IMSubtract : IMBinary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return lhs.Eval(context) - rhs.Eval(context); }
    }
    class IMMultiply : IMBinary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return lhs.Eval(context) * rhs.Eval(context); }
    }
    class IMSaturate : IMUnary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { double v = value.Eval(context); return v < 0 ? 0 : (v < 1 ? v : 1); }
    }
    class IMThreshold : IMUnary<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return value.Eval(context) < 0.5 ? 0 : 1; }
    }
    class IMLerp : IMNNode<double>
    {
        public bool IsConstant() { return value0.IsConstant() && value1.IsConstant() && factor.IsConstant(); }
        public double Eval(EvalContext context) { double f = factor.Eval(context); return value0.Eval(context) * (1 - f) + value1.Eval(context) * f; }
        public IMNNode<double> Value0 { get { return value0; } set { value0 = value; } }
        public IMNNode<double> Value1 { get { return value1; } set { value1 = value; } }
        public IMNNode<double> Factor { get { return factor; } set { factor = value; } }
        protected IMNNode<double> value0, value1, factor; 
    }
    class IMBrickMask : IMSample2D<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return TextureBrick.BrickMask(u.Eval(context), v.Eval(context)); }
    }
    class IMBrickNoise : IMSample2D<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return TextureBrick.BrickNoise(u.Eval(context), v.Eval(context)); }
    }
    class IMPerlin2D : IMSample2D<double>, IMNNode<double>
    {
        public double Eval(EvalContext context) { return TexturePerlin.PerlinNoise2D(u.Eval(context), v.Eval(context)); }
    }
}