using System.Diagnostics;
using System.Linq.Expressions;
using System.Reflection;

namespace RenderToy.Expressions
{
    public static partial class ExpressionExtensions
    {
        public static ExpressionFlatten<TDelegate> Flatten<TDelegate>(this Expression<TDelegate> func)
        {
            return new ExpressionFlatten<TDelegate>(func);
        }
    }
    public class ExpressionFlatten<TDelegate>
    {
        public ExpressionFlatten(Expression<TDelegate> func)
        {
            Original = func;
            Replaced = ExpressionReplaceCall.Replace(func);
            Call = Replaced.Compile();
        }
        public readonly Expression<TDelegate> Original;
        public readonly Expression<TDelegate> Replaced;
        public readonly TDelegate Call;
        class ExpressionReplaceCall : ExpressionVisitor
        {
            public static Expression<TDelegate> Replace(Expression<TDelegate> expression)
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
            public static Expression<TDelegate> Replace(Expression<TDelegate> expression, string name)
            {
                return (Expression<TDelegate>)Expression.Lambda(Replace(expression.Body), name, expression.Parameters);
            }
            static Expression Replace(Expression expression)
            {
                return new ExpressionReplaceCall().Visit(expression);
            }
            protected override Expression VisitMethodCall(MethodCallExpression node)
            {
                var methodinfo = node.Method;
                var membertype = methodinfo.DeclaringType;
                var expressionfield = membertype.GetField("Replaced");
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
                if (node.Expression.NodeType == ExpressionType.Lambda)
                {
                    return base.VisitInvocation(node);
                }
                if (node.Expression.NodeType == ExpressionType.MemberAccess)
                {
                    var membercall = (MemberExpression)node.Expression;
                    if (membercall.Expression.NodeType != ExpressionType.MemberAccess) goto FAIL;
                    var memberhost = (MemberExpression)membercall.Expression;
                    object hostobj = null;
                    if (memberhost.Member is FieldInfo field)
                    {
                        hostobj = field.GetValue(null);
                    }
                    var expressionfield = membercall.Member.DeclaringType.GetField("Replaced");
                    if (expressionfield == null) goto FAIL;
                    var expressiontree = expressionfield.GetValue(hostobj) as LambdaExpression;
                    if (expressiontree == null) goto FAIL;
                    return base.VisitInvocation(Expression.Invoke(expressiontree, node.Arguments));
                }
                FAIL:
                Debug.Assert(false, "Couldn't replace node '" + node + "'.");
                return base.VisitInvocation(node);
            }

        }
    }
}