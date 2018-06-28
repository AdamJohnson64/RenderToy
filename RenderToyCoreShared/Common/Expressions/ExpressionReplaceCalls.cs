////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.Primitives;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;

namespace RenderToy.Expressions
{
    class ExpressionReplaceCalls : ExpressionVisitor
    {
        public static Expression Replace(Expression expression)
        {
            return new ExpressionReplaceCalls().Visit(expression);
        }
        public static Expression<TDelegate> Replace<TDelegate>(Expression<TDelegate> expression)
        {
            string name = null;
            if (expression.Body is InvocationExpression invocation)
            {
                if (!(invocation.Expression is MemberExpression memberaccess)) goto FAIL;
                name = memberaccess.Member.Name;
            }
            if (expression.Body is MethodCallExpression methodcall)
            {
                name = methodcall.Method.Name;
            }
            FAIL:
            return Replace(expression, name);
        }
        public static Expression<TDelegate> Replace<TDelegate>(Expression<TDelegate> expression, string name)
        {
            return (Expression<TDelegate>)Expression.Lambda(Replace(expression.Body), name, expression.Parameters);
        }
        protected override Expression VisitMethodCall(MethodCallExpression node)
        {
            var methodinfo = node.Method;
            var membertype = methodinfo.DeclaringType;
            var expressionname = methodinfo.Name;
            foreach (var param in methodinfo.GetParameters())
            {
                expressionname = expressionname + "_" + param.ParameterType.Name;
            }
            var expressionfield = membertype.GetField(expressionname);
            if (expressionfield == null) goto FAIL;
            var expressiontree = expressionfield.GetValue(null) as LambdaExpression;
            if (expressiontree == null) goto FAIL;
            return base.VisitInvocation(Expression.Invoke(expressiontree, node.Arguments));
            FAIL:
            if (!typeof(IQueryable).IsAssignableFrom(node.Method.ReturnType))
            {
                throw new NotSupportedException();
            }
            Debug.Assert(false, "Couldn't replace node '" + node + "'.");
            return base.VisitMethodCall(node);
        }
        protected override Expression VisitInvocation(InvocationExpression node)
        {
            if (!(node.Expression is MemberExpression member)) goto FAIL;
            var membertype = member.Member.DeclaringType;
            var expressionname = member.Member.Name;
            foreach (var param in node.Arguments)
            {
                expressionname = expressionname + "_" + param.Type.Name;
            }
            var expressionfield = membertype.GetField(expressionname);
            if (expressionfield == null) goto FAIL;
            var expressiontree = expressionfield.GetValue(null) as LambdaExpression;
            if (expressiontree == null) goto FAIL;
            return base.VisitInvocation(Expression.Invoke(expressiontree, node.Arguments));
            FAIL:
            //Debug.Assert(false, "Couldn't replace node '" + node + "'.");
            return base.VisitInvocation(node);
        }
    }
}