////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class MNTexCoordV : ExpressionBase, IMNNode<double>, INamed
    {
        public string Name { get { return "V"; } }
        public bool IsConstant() { return false; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Field(evalcontext, evalcontext.Type.GetField("V"));
        }
    }

}