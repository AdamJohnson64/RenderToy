﻿using RenderToy.Expressions;
using RenderToy.Utility;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class MNSaturate : MNUnary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Saturate"; } }
        public static Expression CreateSaturate(Expression v)
        {
            return Saturate.Replaced.CreateInvoke(v);
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            return CreateSaturate(value.CreateExpression(evalcontext));
        }
    }
}