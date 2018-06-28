////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Linq.Expressions;

namespace RenderToy.Expressions
{
    public static class ExpressionReducer
    {
        public static Expression<TDelegate> Reduce<TDelegate>(Expression<TDelegate> expression)
        {
            return expression.Update(Reduce(expression.Body), expression.Parameters);
        }
        public static Expression<TDelegate> Rename<TDelegate>(Expression<TDelegate> expression, string name)
        {
            return (Expression<TDelegate>)Expression.Lambda(expression.Body, name, expression.Parameters);
        }
        public static Expression Reduce(Expression expression)
        {
            var termcounter = new ExpressionCounter();
            termcounter.Visit(expression);
            var replace = termcounter.Found.Where(i => i.Value > 1).Select((x, i) => new KeyValuePair<Expression, Expression>(x.Key, Expression.Parameter(x.Key.Type, "TEMP" + i))).ToArray();
            if (replace.Length == 0) return expression;
            var termreplacer = new ExpressionSubstitution(replace);
            return Expression.Block(
                replace.Select(i => i.Value).OfType<ParameterExpression>(),
                replace.Where(i => i.Value is ParameterExpression).Select(i => Expression.Assign(i.Value, i.Key)).Concat(new[] { termreplacer.Visit(expression) }));
        }
        public static InvocationExpression CreateInvoke<T, TResult>(this Expression<Func<T, TResult>> func, Expression a)
        {
            return Expression.Invoke(func, a);
        }
        public static InvocationExpression CreateInvoke<T1, T2, TResult>(this Expression<Func<T1, T2, TResult>> func, Expression a, Expression b)
        {
            return Expression.Invoke(func, a, b);
        }
        static Expression<Func<T, TResult>> CreateLambda<T, TResult>(this Expression<Func<T, TResult>> func, string name, ParameterExpression a)
        {
            Debug.Assert(name != null);
            return (Expression<Func<T, TResult>>)Expression.Lambda(func.Body, name, new[] { a });
        }
        static Expression<Func<T1, T2, TResult>> CreateLambda<T1, T2, TResult>(this Expression<Func<T1, T2, TResult>> func, string name, ParameterExpression a, ParameterExpression b)
        {
            Debug.Assert(name != null);
            return (Expression<Func<T1, T2, TResult>>)Expression.Lambda(func.Body, name, new[] { a, b });
        }
        public static Expression<Func<T1, T2, TResult>> CreateLambda<T1, T2, TResult>(this Expression<Func<T1, T2, TResult>> func)
        {
            var a = Expression.Parameter(typeof(T1), "a");
            var b = Expression.Parameter(typeof(T2), "b");
            return (Expression<Func<T1, T2, TResult>>)Expression.Lambda(func.Body, a, b);
        }
        public static Expression<Func<T1, T2, TResult>> CreateLambda<T1, T2, TResult>(this Func<ParameterExpression, ParameterExpression, Expression<Func<T1, T2, TResult>>> func, ParameterExpression a, ParameterExpression b)
        {
            return CreateLambda(func(a, b), "BLAH", a, b);
        }
        public static Expression<Func<T1, T2, TResult>> CreateLambda<T1, T2, TResult>(this Func<ParameterExpression, ParameterExpression, Expression<Func<T1, T2, TResult>>> func)
        {
            var a = Expression.Parameter(typeof(T1), "a");
            var b = Expression.Parameter(typeof(T2), "b");
            return CreateLambda(func(a, b), func.Method.Name, a, b);
        }
    }
}