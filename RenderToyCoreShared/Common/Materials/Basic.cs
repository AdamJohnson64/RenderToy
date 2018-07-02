////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Expressions;
using RenderToy.Math;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public class ExpressionBase
    {
        static readonly Expression<Func<double, double>> FloorFn = (d) => System.Math.Floor(d);
        public static readonly ExpressionFlatten<Func<double, double>> Floor = FloorFn.Rename("Floor").Flatten();
        static readonly Expression<Func<double, double, double, double>> LerpFn = (a, b, x) => a * (1 - x) + b * x;
        public static readonly ExpressionFlatten<Func<double, double, double, double>> Lerp = LerpFn.CommonSubexpression().Rename("Lerp").Flatten();
        static readonly Expression<Func<double, double, double>> PowFn = (mantissa, exponent) => System.Math.Pow(mantissa, exponent);
        public static readonly ExpressionFlatten<Func<double, double, double>> Pow = PowFn.Rename("Pow").Flatten();
        static readonly Expression<Func<double, double>> SaturateFn = (f) => f < 0 ? 0 : (f < 1 ? f : 1);
        public static readonly ExpressionFlatten<Func<double, double>> Saturate = SaturateFn.CommonSubexpression().Rename("Saturate").Flatten();
        static readonly Expression<Func<double, double>> SquareFn = d => d * d;
        public static readonly ExpressionFlatten<Func<double, double>> Square = SquareFn.CommonSubexpression().Rename("Square").Flatten();
        static readonly Expression<Func<double, double>> SqrtFn = d => System.Math.Sqrt(d);
        public static readonly ExpressionFlatten<Func<double, double>> Sqrt = SqrtFn.Rename("Sqrt").Flatten();
        static readonly Expression<Func<double, double>> TileFn = (f) => f - System.Math.Floor(f);
        public static readonly ExpressionFlatten<Func<double, double>> Tile = TileFn.CommonSubexpression().Rename("Tile").Flatten();
    }
    public interface IMaterial
    {
        bool IsConstant();
    }
    public class EvalContext
    {
        public double U, V;
        public EvalContext()
        {
            U = 0;
            V = 0;
        }
        public EvalContext(EvalContext clonefrom)
        {
            U = clonefrom.U;
            V = clonefrom.V;
        }
    }
    public interface IMNNode : IMaterial
    {
        Expression CreateExpression(Expression evalcontext);
    }
    public interface IMNNode<T> : IMNNode
    {
    }
    public abstract class MNUnary<T> : ExpressionBase
    {
        public bool IsConstant() { return value.IsConstant(); }
        public IMNNode<T> Value { get { return value; } set { this.value = value; } }
        protected IMNNode<T> value;
    }
    public abstract class MNBinary<T> : ExpressionBase
    {
        public bool IsConstant() { return lhs.IsConstant() && rhs.IsConstant(); }
        public IMNNode<T> Lhs { get { return lhs; } set { lhs = value; } }
        public IMNNode<T> Rhs { get { return rhs; } set { rhs = value; } }
        protected IMNNode<T> lhs, rhs;
    }
    public abstract class MNSample2D<T> : ExpressionBase
    {
        public bool IsConstant() { return u.IsConstant() && v.IsConstant(); }
        public IMNNode<double> U { get { return u; } set { u = value; } }
        public IMNNode<double> V { get { return v; } set { v = value; } }
        protected IMNNode<double> u, v;
    }
    public class MNTexCoordU : ExpressionBase, IMNNode<double>, INamed
    {
        public string Name { get { return "U"; } }
        public bool IsConstant() { return false; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Field(evalcontext, evalcontext.Type.GetField("U"));
        }
    }
    public sealed class MNTexCoordV : IMNNode<double>, INamed
    {
        public string Name { get { return "V"; } }
        public bool IsConstant() { return false; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Field(evalcontext, evalcontext.Type.GetField("V"));
        }
    }
    public sealed class MNConstant : IMNNode<double>, INamed
    {
        public string Name { get { return value.ToString(); } }
        public bool IsConstant() { return true; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Constant(value);
        }
        public double Value { get { return value; } set { this.value = value; } }
        double value;
    }
    public sealed class MNVector4D : IMNNode<Vector4D>, INamed
    {
        public string Name { get { return "RGBA"; } }
        public bool IsConstant() { return r.IsConstant() && g.IsConstant() && b.IsConstant() && a.IsConstant(); }
        public Expression CreateExpression(Expression evalcontext)
        {
            var parts = new IMNNode<double>[] { r, g, b, a }.Distinct().Select((v,i) => new { Node = v, Index = i }).ToArray();
            Dictionary<IMNNode<double>, Expression> lookup = null;
            if (parts.Length < 4)
            {
                lookup = parts.ToDictionary(k => k.Node, v => (Expression)Expression.Variable(typeof(double), "V4Part" + v.Index));
            }
            else
            {
                lookup = parts.ToDictionary(k => k.Node, v => v.Node.CreateExpression(evalcontext));
            }
            Expression interior = Expression.New(
                typeof(Vector4D).GetConstructor(new System.Type[] { typeof(double), typeof(double), typeof(double), typeof(double) }),
                new Expression[] { lookup[r], lookup[g], lookup[b], lookup[a] });
            if (parts.Length < 4)
            {
                interior = Expression.Block(
                    lookup.Select(i => i.Value).OfType<ParameterExpression>().ToArray(),
                    lookup.Select(i => Expression.Assign(i.Value, i.Key.CreateExpression(evalcontext))).Concat(new Expression[] { interior }).ToArray());

            }
            return interior;
        }
        public IMNNode<double> R { get { return r; } set { r = value; } }
        public IMNNode<double> G { get { return g; } set { g = value; } }
        public IMNNode<double> B { get { return b; } set { b = value; } }
        public IMNNode<double> A { get { return a; } set { a = value; } }
        IMNNode<double> r, g, b, a;
    }
    public sealed class MNAdd : MNBinary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "+"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Add(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
    public sealed class MNSubtract : MNBinary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "-"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Subtract(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
    public sealed class MNMultiply : MNBinary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "X"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Multiply(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
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
    sealed class MNSaturate : MNUnary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Saturate"; } }
        public static Expression CreateSaturate(Expression v)
        {
            return Saturate.Replaced.CreateInvoke(v);
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            return CreateSaturate(value.CreateExpression(evalcontext));
        }
    }
    public sealed class MNSin : MNUnary<double>, IMNNode<double>, INamed
    {
        static Expression<Func<double, double>> SinFn2 = (f) => System.Math.Sin(f);
        static Expression<Func<double, double>> SinFn = SinFn2.Rename("Sin");
        public string Name { get { return "Sin"; } }
        public Expression CreateExpression(Expression evalcontext) { return Expression.Invoke(SinFn, Value.CreateExpression(evalcontext)); }
    }
    public sealed class MNThreshold : MNUnary<double>, IMNNode<double>, INamed
    {
        static Expression<Func<double, double>> ThresholdFn2 = (f) => f < 0.5 ? 0 : 1;
        static Expression<Func<double, double>> ThresholdFn = ThresholdFn2.Rename("Threshold");
        public string Name { get { return "Threshold"; } }
        public Expression CreateExpression(Expression evalcontext) { return Expression.Invoke(ThresholdFn, Value.CreateExpression(evalcontext)); }
    }
    public sealed class MNLerp : ExpressionBase, IMNNode<double>, INamed
    {
        public string Name { get { return "Lerp"; } }
        public bool IsConstant() { return value0.IsConstant() && value1.IsConstant() && factor.IsConstant(); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Lerp.Replaced.CreateInvoke(value0.CreateExpression(evalcontext), value1.CreateExpression(evalcontext), factor.CreateExpression(evalcontext));
        }
        public IMNNode<double> Value0 { get { return value0; } set { value0 = value; } }
        public IMNNode<double> Value1 { get { return value1; } set { value1 = value; } }
        public IMNNode<double> Factor { get { return factor; } set { factor = value; } }
        IMNNode<double> value0, value1, factor;
    }
    public sealed class GenericMaterial : IMNNode<Vector4D>, INamed
    {
        public GenericMaterial(string name, Vector4D ambient, Vector4D diffuse, Vector4D specular, Vector4D reflect, Vector4D refract, double ior)
        {
            this.name = name;
            Ambient = ambient;
            Diffuse = diffuse;
            Specular = specular;
            Reflect = reflect;
            Refract = refract;
            Ior = ior;
        }
        public string Name
        {
            get
            { 
                return name;
            }
        }
        public bool IsConstant()
        {
            return true;
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Constant(Diffuse);
        }
        public readonly string name;
        public readonly Vector4D Ambient;
        public readonly Vector4D Diffuse;
        public readonly Vector4D Specular;
        public readonly Vector4D Reflect;
        public readonly Vector4D Refract;
        public readonly double Ior;
    }
}