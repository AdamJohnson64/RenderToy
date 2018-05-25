using RenderToy.Materials;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Linq.Expressions;

namespace RenderToy
{
    class VisitorComparator : IEqualityComparer<Expression>
    {
        public static int Complexity(Expression expression)
        {
            if (expression == null) return 0;
            if (expression is BinaryExpression)
            {
                var cast = (BinaryExpression)expression;
                return 1 + Complexity(cast.Left) + Complexity(cast.Right);
            }
            else if (expression is BlockExpression)
            {
                var cast = (BlockExpression)expression;
                return 1 + cast.Expressions.Concat(cast.Variables).Select(i => Complexity(i)).Sum();
            }
            else if (expression is ConditionalExpression)
            {
                var cast = (ConditionalExpression)expression;
                return 1 + Complexity(cast.Test) + Complexity(cast.IfTrue) + Complexity(cast.IfFalse);
            }
            else if (expression is ConstantExpression)
            {
                var cast = (ConstantExpression)expression;
                return 1;
            }
            else if (expression.NodeType == ExpressionType.Convert)
            {
                var cast = (UnaryExpression)expression;
                return 1 + Complexity(cast.Operand);
            }
            else if (expression is MemberExpression)
            {
                var cast = (MemberExpression)expression;
                return 1 + Complexity(cast.Expression);
            }
            else if (expression is MethodCallExpression)
            {
                var cast = (MethodCallExpression)expression;
                return 100 + cast.Arguments.Select(i => Complexity(i)).Sum();
            }
            else if (expression is NewExpression)
            {
                var cast = (NewExpression)expression;
                return 1 + cast.Arguments.Select(i => Complexity(i)).Sum();
            }
            else if (expression is ParameterExpression)
            {
                var cast = (ParameterExpression)expression;
                return 1;
            }
            else
            {
                throw new NotSupportedException();
            }
        }
        public static bool MyEquals(Expression lhs2, Expression rhs2)
        {
            if (lhs2 == null && rhs2 == null) return true;
            var lhs = lhs2 as Expression;
            var rhs = rhs2 as Expression;
            if (lhs == null || rhs == null) return false;
            if (lhs.NodeType != rhs.NodeType) return false;
            var type = lhs.NodeType;
            if (lhs is BinaryExpression && rhs is BinaryExpression && lhs.GetType() == rhs.GetType())
            {
                var cast1 = (BinaryExpression)lhs;
                var cast2 = (BinaryExpression)rhs;
                if (!(cast1.Left.NodeType == cast2.Left.NodeType && cast1.Right.NodeType == cast2.Right.NodeType)) return false;
                if (!(MyEquals(cast1.Left, cast2.Left) && MyEquals(cast1.Right, cast2.Right))) return false;
                if (cast1.Type != cast2.Type) return false;
                return true;
            }
            else if (type == ExpressionType.Block)
            {
                var cast1 = (BlockExpression)lhs;
                var cast2 = (BlockExpression)rhs;
                if (cast1.Expressions.Count != cast2.Expressions.Count) return false;
                for (int i = 0; i < cast1.Expressions.Count; ++i)
                {
                    if (!MyEquals(cast1.Expressions[i], cast2.Expressions[i])) return false;
                }
                if (cast1.Variables.Count != cast2.Variables.Count) return false;
                for (int i = 0; i < cast1.Variables.Count; ++i)
                {
                    if (!MyEquals(cast1.Variables[i], cast2.Variables[i])) return false;
                }
                return true;
            }
            else if (type == ExpressionType.Conditional)
            {
                var cast1 = (ConditionalExpression)lhs;
                var cast2 = (ConditionalExpression)rhs;
                if (!MyEquals(cast1.IfTrue, cast2.IfTrue)) return false;
                if (!MyEquals(cast1.IfFalse, cast2.IfFalse)) return false;
                if (!MyEquals(cast1.Test, cast2.Test)) return false;
                return true;
            }
            else if (type == ExpressionType.Constant)
            {
                if (!object.Equals(((ConstantExpression)lhs).Value, ((ConstantExpression)rhs).Value)) return false;
                return true;
            }
            else if (type == ExpressionType.Convert)
            {
                var cast1 = (UnaryExpression)lhs;
                var cast2 = (UnaryExpression)rhs;
                if (cast1.Method != cast2.Method) return false;
                if (!MyEquals(cast1.Operand, cast2.Operand)) return false;
                return true;
            }
            else if (type == ExpressionType.Call)
            {
                var cast1 = (MethodCallExpression)lhs;
                var cast2 = (MethodCallExpression)rhs;
                if (!Equals(cast1.Method, cast2.Method)) return false;
                if (cast1.Arguments.Count != cast2.Arguments.Count) return false;
                for (int i = 0; i < cast1.Arguments.Count; ++i)
                {
                    if (!MyEquals(cast1.Arguments[i], cast2.Arguments[i])) return false;
                }
                return true;
            }
            else if (type == ExpressionType.MemberAccess)
            {
                var cast1 = (MemberExpression)lhs;
                var cast2 = (MemberExpression)rhs;
                if (cast1.Member != cast2.Member) return false;
                if (!MyEquals(cast1.Expression, cast2.Expression)) return false;
                return true;
            }
            else if (type == ExpressionType.New)
            {
                var cast1 = (NewExpression)lhs;
                var cast2 = (NewExpression)rhs;
                if (cast1.Constructor != cast2.Constructor) return false;
                if (cast1.Arguments.Count != cast2.Arguments.Count) return false;
                for (int i = 0; i < cast1.Arguments.Count; ++i)
                {
                    if (!MyEquals(cast1.Arguments[i], cast2.Arguments[i])) return false;
                }
                return true;
            }
            else if (type == ExpressionType.Parameter)
            {
                var cast1 = (ParameterExpression)lhs;
                var cast2 = (ParameterExpression)rhs;
                if (cast1 != cast2) return false;
                return true;
            }
            else
            {
                return object.Equals(lhs, rhs);
            }
        }
        public bool Equals(Expression x, Expression y)
        {
            return MyEquals(x, y);
        }
        public int GetHashCode(Expression obj)
        {
            return obj.GetType().GetHashCode();
        }
    }
    class VisitorExpand : ExpressionVisitor
    {
        protected override Expression VisitBlock(BlockExpression node)
        {
            if (node == null) return base.VisitBlock(node);
            for (int iter = 0; iter < 100; ++iter)
            {
                var assign = node.Expressions.Where(i => i.NodeType == ExpressionType.Assign).OfType<BinaryExpression>().FirstOrDefault();
                if (assign == null) break;
                node = Expression.Block(
                    node.Type,
                    node.Variables.Where(i => !VisitorComparator.MyEquals(i, assign.Left)).ToArray(),
                    node.Expressions.Where(i => !VisitorComparator.MyEquals(i, assign)).Select(i => new VisitorReplace(assign.Left, assign.Right).Visit(i)).ToArray());
            }
            return node.Expressions.Count == 1 ? base.Visit(node.Expressions[0]) : base.VisitBlock(node);
        }
    }
    class VisitorReplace : ExpressionVisitor
    {
        public VisitorReplace(Expression oldexpression, Expression newexpression)
        {
            _oldexpression = oldexpression;
            _newexpression = newexpression;
        }
        public override Expression Visit(Expression node)
        {
            if (node == null) return base.Visit(node);
            return VisitorComparator.MyEquals(node, _oldexpression) ? base.Visit(_newexpression) : base.Visit(node);
        }
        Expression _oldexpression;
        Expression _newexpression;
    }
    class VisitorTest : ExpressionVisitor
    {
        public override Expression Visit(Expression node)
        {
            if (node != null)
            {
                if (countup.ContainsKey(node))
                {
                    countup[node] = countup[node] + 1;
                }
                else
                {
                    countup[node] = 1;
                }
            }
            if (node is ParameterExpression)
            {
                found.Add(node);
                return base.Visit(node);
            }
            var find = found.FirstOrDefault(i => VisitorComparator.MyEquals(node, i));
            if (find != null) return find;
            found.Add(node);
            return base.Visit(node);
        }
        class VariableBinding
        {
            public ParameterExpression Variable;
            public Expression Value;
        }
        public static Expression Reduce(Expression expression)
        {
            expression = new VisitorExpand().Visit(expression);
            var compare1 = expression;
            var bindthese = new List<VariableBinding>();
            for (int i = 0; i < 8; ++i)
            {
                var visitor1 = new VisitorTest();
                expression = visitor1.Visit(expression);
                var ranking =
                    visitor1.countup
                    .Where(n => n.Value > 1)
                    .Where(n => VisitorComparator.Complexity(n.Key) > 10)
                    .OrderByDescending(n => n.Value)
                    .ThenByDescending(n => VisitorComparator.Complexity(n.Key));
                if (ranking.Count() == 0) break;
                var worst = ranking.FirstOrDefault();
                var subst = Expression.Variable(worst.Key.Type, "PASS" + i);
                var visitor2 = new VisitorReplace(worst.Key, subst);
                var inner = visitor2.Visit(expression);
                expression = inner;
                bindthese.Add(new VariableBinding { Variable = subst, Value = worst.Key });
            }
            expression = Expression.Block(
                expression.Type,
                bindthese.Select(i => i.Variable).ToArray(),
                bindthese.Select(i => Expression.Assign(i.Variable, i.Value)).Concat(new Expression[] { expression }).ToArray()
                );
            //expression = new VisitorExpand().Visit(expression);
            //var compare2 = expression;
            //var iseq = VisitorComparator.MyEquals(compare1, compare2);
            return expression;
        }
        public static void Test()
        {
            var material = StockMaterials.Brick;
            var param = Expression.Parameter(typeof(EvalContext), "EvalContext");
            var body = material.CreateExpression(param);
            var original = body;
            var lambdaoriginal = Expression.Lambda<Func<EvalContext, Vector4D>>(original, param).Compile();
            var reduced = Reduce(body);
            var lambdareduced = Expression.Lambda<Func<EvalContext, Vector4D>>(reduced, param).Compile();

            var evalcontext = new EvalContext { U = 0.5, V = 0.5 };
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            for (int i = 0; i < 10000; ++i)
            {
                var evalnormal = material.Eval(evalcontext);
            }
            stopwatch.Stop();
            Debug.WriteLine("Classic Took " + stopwatch.ElapsedMilliseconds);
            stopwatch.Restart();
            for (int i = 0; i < 10000; ++i)
            {
                var evallambda = lambdaoriginal(evalcontext);
            }
            stopwatch.Stop();
            Debug.WriteLine("Lambda Original Took " + stopwatch.ElapsedMilliseconds);
            stopwatch.Restart();
            for (int i = 0; i < 10000; ++i)
            {
                var evallambda = lambdareduced(evalcontext);
            }
            stopwatch.Stop();
            Debug.WriteLine("Lambda Reduced Took " + stopwatch.ElapsedMilliseconds);
            for (int y = 0; y < 100; ++y)
            {
                for (int x = 0; x < 100; ++x)
                {
                    evalcontext.U = (x + 0.5) / 100.0;
                    evalcontext.V = (y + 0.5) / 100.0;
                    var test1 = material.Eval(evalcontext);
                    var test2 = lambdaoriginal(evalcontext);
                    var test3 = lambdareduced(evalcontext);
                    int test = 0;
                }
            }
        }
        List<Expression> found = new List<Expression>();
        Dictionary<Expression, int> countup = new Dictionary<Expression, int>(new VisitorComparator());
    }
}