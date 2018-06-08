////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
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
    }
}