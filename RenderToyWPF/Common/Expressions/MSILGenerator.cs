﻿using RenderToy.Materials;
using System;
using System.Linq.Expressions;

namespace RenderToy.Expressions
{
    public static class MSILExtensions
    {
        public static Expression GenerateMSIL(this IMNNode material)
        {
            var expression = material.CreateExpression(_evalcontext);
            return expression;
        }
        public static Func<EvalContext,T> CompileMSIL<T>(this IMNNode<T> material)
        {
            return Expression.Lambda<Func<EvalContext, T>>(material.GenerateMSIL(), _evalcontext).Compile();
        }
        static ParameterExpression _evalcontext = Expression.Parameter(typeof(EvalContext));
    }
}