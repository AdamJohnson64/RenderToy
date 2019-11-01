////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Expressions;
using RenderToy.Utility;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public sealed class MNPower : ExpressionBase, IMNNode<double>, INamed
    {
        public string Name { get { return "Power"; } }
        public bool IsConstant() { return value.IsConstant() && exponent.IsConstant(); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Pow.Replaced.CreateInvoke(Value.CreateExpression(evalcontext), Exponent.CreateExpression(evalcontext));
        }
        public IMNNode<double> Value { get { return this.value; } set { this.value = value; } }
        public IMNNode<double> Exponent { get { return exponent; } set { exponent = value; } }
        IMNNode<double> value, exponent;
    }
}