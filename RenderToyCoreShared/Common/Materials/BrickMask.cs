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
    sealed class BrickMask : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Brick Mask"; } }
        static Expression<Func<double, double, double>> TempFn2 = (u, v) => (v < MortarWidth) ? 0 : (((v < 0.5 - MortarWidth) ? ((u < MortarWidth) ? 0 : ((u < 1.0 - MortarWidth) ? 1 : 0)) : (v < 0.5 + MortarWidth) ? 0 : ((v < 1.0 - MortarWidth) ? (u < 0.5 - MortarWidth) ? 1 : ((u < 0.5 + MortarWidth) ? 0 : 1) : 0)));
        static Expression<Func<double, double, double>> TempFn = ExpressionReducer.Rename(TempFn2, "BrickMask2D");
        public Expression CreateExpression(Expression evalcontext) => TempFn.CreateInvoke(TileFn.CreateInvoke(u.CreateExpression(evalcontext)), TileFn.CreateInvoke(v.CreateExpression(evalcontext)));
        const double MortarWidth = 0.025;
    }
}