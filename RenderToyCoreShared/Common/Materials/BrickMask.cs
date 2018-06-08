////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    sealed class BrickMask : MNSample2D<double>, IMNNode<double>, INamed
    {
        const double MortarWidth = 0.025;
        public static Expression<Func<double, double, double>> Temp = (u, v) => (v < MortarWidth) ? 0 : (((v < 0.5 - MortarWidth) ? ((u < MortarWidth) ? 0 : ((u < 1.0 - MortarWidth) ? 1 : 0)) : (v < 0.5 + MortarWidth) ? 0 : ((v < 1.0 - MortarWidth) ? (u < 0.5 - MortarWidth) ? 1 : ((u < 0.5 + MortarWidth) ? 0 : 1) : 0)));
        public static Func<double, double, double> CallTemp = Temp.Compile();
        public static double Compute(double u, double v)
        {
            return CallTemp(u - System.Math.Floor(u), v - System.Math.Floor(v));
        }
        public string Name { get { return "Brick Mask"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Invoke(Temp,
                Expression.Invoke(TileFn, u.CreateExpression(evalcontext)),
                Expression.Invoke(TileFn, v.CreateExpression(evalcontext)));
        }
    }
}