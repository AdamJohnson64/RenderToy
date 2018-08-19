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
    public sealed class MNThreshold : MNUnary<double>, IMNNode<double>, INamed
    {
        static Expression<Func<double, double>> ThresholdFn2 = (f) => f < 0.5 ? 0 : 1;
        static Expression<Func<double, double>> ThresholdFn = ThresholdFn2.Rename("Threshold");
        public string Name { get { return "Threshold"; } }
        public Expression CreateExpression(Expression evalcontext) { return Expression.Invoke(ThresholdFn, Value.CreateExpression(evalcontext)); }
    }
}