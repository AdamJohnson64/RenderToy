using System;
using System.Diagnostics;
using System.Linq.Expressions;

namespace RenderToy.Expressions
{
    public static partial class ExpressionExtensions
    {
        public static InvocationExpression CreateInvoke<T, TResult>(this Expression<Func<T, TResult>> func, Expression a)
        {
            return Expression.Invoke(func, a);
        }
        public static InvocationExpression CreateInvoke<T1, T2, TResult>(this Expression<Func<T1, T2, TResult>> func, Expression a, Expression b)
        {
            return Expression.Invoke(func, a, b);
        }
        public static InvocationExpression CreateInvoke<T1, T2, T3, TResult>(this Expression<Func<T1, T2, T3, TResult>> func, Expression a, Expression b, Expression c)
        {
            return Expression.Invoke(func, a, b, c);
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
            return (Expression<Func<T1, T2, TResult>>)Expression.Lambda(func.Body, func.Parameters[0], func.Parameters[1]);
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
        public static Expression<TDelegate> Rename<TDelegate>(this Expression<TDelegate> expression, string name)
        {
            return (Expression<TDelegate>)Expression.Lambda(expression.Body, name, expression.Parameters);
        }
    }
}