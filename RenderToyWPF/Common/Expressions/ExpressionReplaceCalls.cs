using System.Linq.Expressions;

namespace RenderToy.Expressions
{
    public static partial class ExpressionExtensions
    {
        public static Expression ReplaceCalls(this Expression expression)
        {
            return new ExpressionReplaceCalls().Visit(expression);
        }
        public static Expression<TDelegate> ReplaceCalls<TDelegate>(this Expression<TDelegate> expression)
        {
            return (Expression<TDelegate>)Expression.Lambda(ReplaceCalls(expression.Body), expression.Parameters);
        }
        class ExpressionReplaceCalls : ExpressionVisitor
        {
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
                //if (!typeof(IQueryable).IsAssignableFrom(node.Method.ReturnType))
                //{
                //    throw new NotSupportedException();
                //}
                //Debug.Assert(false, "Couldn't replace node '" + node + "'.");
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
}