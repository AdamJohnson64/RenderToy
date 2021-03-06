﻿using System.Collections.Generic;
using System.Linq.Expressions;

namespace RenderToy.Expressions
{
    class ExpressionCounter : ExpressionVisitor
    {
        public override Expression Visit(Expression node)
        {
            if (node != null)
            {
                if (!Found.ContainsKey(node)) Found[node] = 0;
                Found[node] += 1;
            }
            return base.Visit(node);
        }
        public Dictionary<Expression, int> Found = new Dictionary<Expression, int>();
    }
}