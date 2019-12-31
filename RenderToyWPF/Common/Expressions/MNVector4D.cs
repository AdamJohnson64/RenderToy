using RenderToy.Math;
using RenderToy.Utility;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class MNVector4D : IMNNode<Vector4D>, INamed
    {
        public string Name { get { return "RGBA"; } }
        public bool IsConstant() { return r.IsConstant() && g.IsConstant() && b.IsConstant() && a.IsConstant(); }
        public Expression CreateExpression(Expression evalcontext)
        {
            var parts = new IMNNode<double>[] { r, g, b, a }.Distinct().Select((v, i) => new { Node = v, Index = i }).ToArray();
            Dictionary<IMNNode<double>, Expression> lookup = null;
            if (parts.Length < 4)
            {
                lookup = parts.ToDictionary(k => k.Node, v => (Expression)Expression.Variable(typeof(double), "V4Part" + v.Index));
            }
            else
            {
                lookup = parts.ToDictionary(k => k.Node, v => v.Node.CreateExpression(evalcontext));
            }
            Expression interior = Expression.New(
                typeof(Vector4D).GetConstructor(new System.Type[] { typeof(double), typeof(double), typeof(double), typeof(double) }),
                new Expression[] { lookup[r], lookup[g], lookup[b], lookup[a] });
            if (parts.Length < 4)
            {
                interior = Expression.Block(
                    lookup.Select(i => i.Value).OfType<ParameterExpression>().ToArray(),
                    lookup.Select(i => Expression.Assign(i.Value, i.Key.CreateExpression(evalcontext))).Concat(new Expression[] { interior }).ToArray());

            }
            return interior;
        }
        public IMNNode<double> R { get { return r; } set { r = value; } }
        public IMNNode<double> G { get { return g; } set { g = value; } }
        public IMNNode<double> B { get { return b; } set { b = value; } }
        public IMNNode<double> A { get { return a; } set { a = value; } }
        IMNNode<double> r, g, b, a;
    }
}