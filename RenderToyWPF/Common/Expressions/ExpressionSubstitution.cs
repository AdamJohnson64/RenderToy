////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace RenderToy.Expressions
{
    class ExpressionSubstitution : ExpressionVisitor
    {
        public ExpressionSubstitution(IEnumerable<KeyValuePair<Expression, Expression>> replacements)
        {
            Replacements = replacements.ToDictionary(k => k.Key, v => v.Value);
        }
        public override Expression Visit(Expression node)
        {
            if (node != null && Replacements.ContainsKey(node)) return Replacements[node];
            return base.Visit(node);
        }
        public Dictionary<Expression, Expression> Replacements;
    }
}