////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace RenderToy.Expressions
{
    public class ExpressionComparator : IEqualityComparer<Expression>
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
}