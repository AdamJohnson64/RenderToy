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
    public sealed class MNSin : MNUnary<double>, IMNNode<double>, INamed
    {
        static Expression<Func<double, double>> SinFn2 = (f) => System.Math.Sin(f);
        static Expression<Func<double, double>> SinFn = SinFn2.Rename("Sin");
        public string Name { get { return "Sin"; } }
        public Expression CreateExpression(Expression evalcontext) { return Expression.Invoke(SinFn, Value.CreateExpression(evalcontext)); }
    }
}