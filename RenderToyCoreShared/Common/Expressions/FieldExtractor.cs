////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Linq.Expressions;

namespace RenderToy.Expressions
{
    class FieldExtractor : ExpressionVisitor
    {
        public override System.Linq.Expressions.Expression Visit(System.Linq.Expressions.Expression node)
        {
            if (node != null && node.NodeType == ExpressionType.MemberAccess)
            {
                Found.Add(((MemberExpression)node).Member.Name);
            }
            return base.Visit(node);
        }
        public HashSet<string> Found = new HashSet<string>();
    }
}