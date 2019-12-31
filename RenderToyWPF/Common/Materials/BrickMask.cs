using RenderToy.Expressions;
using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class BrickMask : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Brick Mask"; } }
        static Expression<Func<double, double, double>> TempFn = (u, v) => (v < MortarWidth) ? 0 : (((v < 0.5 - MortarWidth) ? ((u < MortarWidth) ? 0 : ((u < 1.0 - MortarWidth) ? 1 : 0)) : (v < 0.5 + MortarWidth) ? 0 : ((v < 1.0 - MortarWidth) ? (u < 0.5 - MortarWidth) ? 1 : ((u < 0.5 + MortarWidth) ? 0 : 1) : 0)));
        static ExpressionFlatten<Func<double, double, double>> Mask = TempFn.Rename("Mask").Flatten();
        public Expression CreateExpression(Expression evalcontext) => Mask.Replaced.CreateInvoke(Tile.Replaced.CreateInvoke(u.CreateExpression(evalcontext)), Tile.Replaced.CreateInvoke(v.CreateExpression(evalcontext)));
        const double MortarWidth = 0.025;
    }
}