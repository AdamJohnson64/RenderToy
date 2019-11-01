////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Expressions;
using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    sealed class Spike : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Spike"; } }
        static Expression<Func<double, double, double>> SpikeFn = (u, v) =>
            Lerp.Call(1, 0, 2 * Saturate.Call(Square.Call(u - 0.5) + Square.Call(v - 0.5)));
        static ExpressionFlatten<Func<double, double, double>> Func = SpikeFn.Rename("Spike").Flatten();
        public Expression CreateExpression(Expression evalcontext) => Func.Replaced.CreateInvoke(u.CreateExpression(evalcontext), v.CreateExpression(evalcontext));
    }
}