////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
    public interface IMaterial
    {
        bool IsConstant();
    }
    public class EvalContext
    {
        public double U, V;
    }
    public interface IMNNode : IMaterial
    {
        Expression CreateExpression(Expression evalcontext);
    }
    public interface IMNNode<T> : IMNNode
    {
        T Eval(EvalContext context);
    }
    abstract class MNUnary<T>
    {
        public bool IsConstant() { return value.IsConstant(); }
        public IMNNode<T> Value { get { return value; } set { this.value = value; } }
        protected IMNNode<T> value;
    }
    abstract class MNBinary<T>
    {
        public bool IsConstant() { return lhs.IsConstant() && rhs.IsConstant(); }
        public IMNNode<T> Lhs { get { return lhs; } set { lhs = value; } }
        public IMNNode<T> Rhs { get { return rhs; } set { rhs = value; } }
        protected IMNNode<T> lhs, rhs;
    }
    abstract class MNSample2D<T>
    {
        public bool IsConstant() { return u.IsConstant() && v.IsConstant(); }
        public IMNNode<double> U { get { return u; } set { u = value; } }
        public IMNNode<double> V { get { return v; } set { v = value; } }
        protected IMNNode<double> u, v;
    }
    class MNTexCoordU : IMNNode<double>, INamed
    {
        public string GetName() { return "U"; }
        public bool IsConstant() { return false; }
        public double Eval(EvalContext context) { return context.U; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.MakeMemberAccess(evalcontext, typeof(EvalContext).GetField("U"));
        }
    }
    class MNTexCoordV : IMNNode<double>, INamed
    {
        public string GetName() { return "V"; }
        public bool IsConstant() { return false; }
        public double Eval(EvalContext context) { return context.V; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.MakeMemberAccess(evalcontext, typeof(EvalContext).GetField("V"));
        }
    }
    class MNConstant : IMNNode<double>, INamed
    {
        public string GetName() { return value.ToString(); }
        public bool IsConstant() { return true; }
        public double Eval(EvalContext context) { return value; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Constant(value);
        }
        public double Value { get { return value; } set { this.value = value; } }
        protected double value;
    }
    class MNVector4D : IMNNode<Vector4D>, INamed
    {
        public string GetName() { return "RGBA"; }
        public bool IsConstant() { return r.IsConstant() && g.IsConstant() && b.IsConstant() && a.IsConstant(); }
        public Vector4D Eval(EvalContext context) { return new Vector4D(r.Eval(context), g.Eval(context), b.Eval(context), a.Eval(context)); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.New(
                typeof(Vector4D).GetConstructor(new System.Type[] { typeof(double), typeof(double), typeof(double), typeof(double) }),
                new Expression[] { R.CreateExpression(evalcontext), G.CreateExpression(evalcontext), B.CreateExpression(evalcontext), A.CreateExpression(evalcontext) });
        }
        public IMNNode<double> R { get { return r; } set { r = value; } }
        public IMNNode<double> G { get { return g; } set { g = value; } }
        public IMNNode<double> B { get { return b; } set { b = value; } }
        public IMNNode<double> A { get { return a; } set { a = value; } }
        protected IMNNode<double> r, g, b, a;
    }
    class MNAdd : MNBinary<double>, IMNNode<double>, INamed
    {
        public string GetName() { return "+"; }
        public double Eval(EvalContext context) { return lhs.Eval(context) + rhs.Eval(context); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Add(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
    class MNSubtract : MNBinary<double>, IMNNode<double>, INamed
    {
        public string GetName() { return "-"; }
        public double Eval(EvalContext context) { return lhs.Eval(context) - rhs.Eval(context); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Subtract(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
    class MNMultiply : MNBinary<double>, IMNNode<double>, INamed
    {
        public string GetName() { return "X"; }
        public double Eval(EvalContext context) { return lhs.Eval(context) * rhs.Eval(context); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Multiply(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
    class MNPower : IMNNode<double>, INamed
    {
        public string GetName() { return "Power"; }
        public bool IsConstant() { return value.IsConstant() && exponent.IsConstant(); }
        public double Eval(EvalContext context) { return Math.Pow(value.Eval(context), exponent.Eval(context)); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Call(
                null,
                typeof(Math).GetMethod("Pow"),
                new Expression[] { Value.CreateExpression(evalcontext), Exponent.CreateExpression(evalcontext) });
        }
        public IMNNode<double> Value { get { return this.value; } set { this.value = value; } }
        public IMNNode<double> Exponent { get { return exponent; } set { exponent = value; } }
        protected IMNNode<double> value, exponent;
    }
    class MNSaturate : MNUnary<double>, IMNNode<double>, INamed
    {
        public string GetName() { return "Saturate"; }
        public double Eval(EvalContext context) { double v = value.Eval(context); return v < 0 ? 0 : (v < 1 ? v : 1); }
        public Expression CreateExpression(Expression evalcontext)
        {
            var temp = Expression.Parameter(typeof(double));
            return Expression.Block(typeof(double), new ParameterExpression[] { temp }, new Expression[] {
                Expression.Assign(temp, value.CreateExpression(evalcontext)),
                Expression.Condition(
                    Expression.LessThan(temp, Expression.Constant(0.0)),
                    Expression.Constant(0.0),
                    Expression.Condition(
                        Expression.LessThan(temp, Expression.Constant(1.0)),
                        temp,
                        Expression.Constant(1.0)))
            });
        }
    }
    class MNSin : MNUnary<double>, IMNNode<double>, INamed
    {
        public string GetName() { return "Sin"; }
        public double Eval(EvalContext context) { return Math.Sin(value.Eval(context)); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Call(
                null,
                typeof(Math).GetMethod("Sin"),
                new Expression[] { Value.CreateExpression(evalcontext) });
        }
    }
    class MNThreshold : MNUnary<double>, IMNNode<double>, INamed
    {
        public string GetName() { return "Threshold"; }
        public double Eval(EvalContext context) { return value.Eval(context) < 0.5 ? 0 : 1; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Condition(
                Expression.LessThan(value.CreateExpression(evalcontext), Expression.Constant(0.5)),
                Expression.Constant(0.0),
                Expression.Constant(1.0));
        }
    }
    class MNLerp : IMNNode<double>, INamed
    {
        public string GetName() { return "Lerp"; }
        public bool IsConstant() { return value0.IsConstant() && value1.IsConstant() && factor.IsConstant(); }
        public double Eval(EvalContext context) { double f = factor.Eval(context); return value0.Eval(context) * (1 - f) + value1.Eval(context) * f; }
        public Expression CreateExpression(Expression evalcontext)
        {
            var temp = Expression.Parameter(typeof(double));
            return Expression.Block(typeof(double), new ParameterExpression[] { temp }, new Expression[]
            {
                Expression.Assign(temp, factor.CreateExpression(evalcontext)),
                Expression.Add(
                    Expression.Multiply(value0.CreateExpression(evalcontext), Expression.Subtract(Expression.Constant(1.0), temp)),
                    Expression.Multiply(value1.CreateExpression(evalcontext), temp)),
            });
        }
        public IMNNode<double> Value0 { get { return value0; } set { value0 = value; } }
        public IMNNode<double> Value1 { get { return value1; } set { value1 = value; } }
        public IMNNode<double> Factor { get { return factor; } set { factor = value; } }
        protected IMNNode<double> value0, value1, factor;
    }
    class BrickMask : MNSample2D<double>, IMNNode<double>, INamed
    {
        public static double Compute(double u, double v)
        {
            const double MortarWidth = 0.025;
            u = u - Math.Floor(u);
            v = v - Math.Floor(v);
            if (v < MortarWidth) return 0;
            else if (v < 0.5 - MortarWidth)
            {
                if (u < MortarWidth) return 0;
                else if (u < 1.0 - MortarWidth) return 1;
                else return 0;
            }
            else if (v < 0.5 + MortarWidth) return 0;
            else if (v < 1.0 - MortarWidth)
            {
                if (u < 0.5 - MortarWidth) return 1;
                else if (u < 0.5 + MortarWidth) return 0;
                else return 1;
            }
            else return 0;
        }
        public string GetName() { return "Brick Mask"; }
        public double Eval(EvalContext context) { return Compute(u.Eval(context), v.Eval(context)); }
        public Expression CreateExpression(Expression evalcontext)
        {
            const double MortarWidth = 0.025;
            var tempu = Expression.Parameter(typeof(double));
            var tempv = Expression.Parameter(typeof(double));
            return Expression.Block(typeof(double),
                new ParameterExpression[] { tempu, tempv },
                new Expression[]
                {
                    Expression.Assign(tempu, u.CreateExpression(evalcontext)),
                    Expression.Assign(tempv, v.CreateExpression(evalcontext)),
                    Expression.Assign(tempu, Expression.Subtract(tempu, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempu }))),
                    Expression.Assign(tempv, Expression.Subtract(tempv, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv }))),
                    Expression.Condition(
                        Expression.LessThan(tempv, Expression.Constant(MortarWidth)),
                        Expression.Constant(0.0),
                        Expression.Condition(
                            Expression.LessThan(tempv, Expression.Subtract(Expression.Constant(0.5), Expression.Constant(MortarWidth))),
                            Expression.Condition(
                                Expression.LessThan(tempu, Expression.Constant(MortarWidth)),
                                Expression.Constant(0.0),
                                Expression.Condition(
                                    Expression.LessThan(tempu, Expression.Subtract(Expression.Constant(1.0), Expression.Constant(MortarWidth))),
                                    Expression.Constant(1.0),
                                    Expression.Constant(0.0))),
                            Expression.Condition(
                                Expression.LessThan(tempv, Expression.Add(Expression.Constant(0.5), Expression.Constant(MortarWidth))),
                                Expression.Constant(0.0),
                                Expression.Condition(
                                    Expression.LessThan(tempv, Expression.Subtract(Expression.Constant(1.0), Expression.Constant(MortarWidth))),
                                    Expression.Condition(
                                        Expression.LessThan(tempu, Expression.Subtract(Expression.Constant(0.5), Expression.Constant(MortarWidth))),
                                        Expression.Constant(1.0),
                                        Expression.Condition(
                                            Expression.LessThan(tempu, Expression.Add(Expression.Constant(0.5), Expression.Constant(MortarWidth))),
                                            Expression.Constant(0.0),
                                            Expression.Constant(1.0))),
                                    Expression.Constant(0.0))))),
                });
        }
    }
    class BrickNoise : MNSample2D<double>, IMNNode<double>, INamed
    {
        public static double Compute(double u, double v)
        {
            if (v - Math.Floor(v) < 0.5)
            {
                u = Math.Floor(u);
                v = Math.Floor(v + 0.5);
            }
            else
            {
                u = Math.Floor(u + 0.5);
                v = Math.Floor(v);
            }
            return Perlin2D.PerlinNoise2D(u * 8, v * 8);
        }
        public string GetName() { return "Brick Noise"; }
        public double Eval(EvalContext context) { return Compute(u.Eval(context), v.Eval(context)); }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double));
            var tempv = Expression.Parameter(typeof(double));
            return Expression.Block(typeof(double),
                new ParameterExpression[] { tempu, tempv },
                new Expression[]
                {
                    Expression.Assign(tempu, u.CreateExpression(evalcontext)),
                    Expression.Assign(tempv, v.CreateExpression(evalcontext)),
                    Expression.Condition(
                        Expression.LessThan(
                            Expression.Subtract(
                                tempv,
                                Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv })),
                                Expression.Constant(0.5)),
                        Expression.Block(new Expression[]
                        {
                            Expression.Assign(tempu, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempu })),
                            Expression.Assign(tempv, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { Expression.Add(tempv, Expression.Constant(0.5)) })),
                        }),
                        Expression.Block(new Expression[]
                        {
                            Expression.Assign(tempu, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { Expression.Add(tempu, Expression.Constant(0.5)) })),
                            Expression.Assign(tempv, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv })),
                        })),
                    Expression.Call(
                        null,
                        typeof(Perlin2D).GetMethod("PerlinNoise2D"),
                        new Expression[]
                        {
                            Expression.Multiply(tempu, Expression.Constant(8.0)),
                            Expression.Multiply(tempv, Expression.Constant(8.0)),
                        })
                });
        }
    }
    class BumpGenerate : IMNNode<Vector4D>, INamed
    {
        public string GetName() { return "Bump Generate"; }
        public bool IsConstant() { return displacement.IsConstant(); }
        public Vector4D Eval(EvalContext context)
        {
            double du1 = displacement.Eval(new EvalContext { U = context.U - 0.001, V = context.V });
            double du2 = displacement.Eval(new EvalContext { U = context.U + 0.001, V = context.V });
            double dv1 = displacement.Eval(new EvalContext { U = context.U, V = context.V - 0.001 });
            double dv2 = displacement.Eval(new EvalContext { U = context.U, V = context.V + 0.001 });
            var normal = MathHelp.Normalized(new Vector3D((du1 - du2) / 0.002, (dv1 - dv2) / 0.002, 1));
            return new Vector4D(normal.X, -normal.Y, 1, 1);
        }
        // TODO: Fill this in.
        public Expression CreateExpression(Expression evalcontext)
        {
            return System.Linq.Expressions.Expression.Call(
                Expression.Constant(this),
                typeof(BumpGenerate).GetMethod("Eval"),
                new Expression[] { evalcontext });
        }
        public IMNNode<double> Displacement
        {
            get { return displacement; }
            set { displacement = value; }
        }
        IMNNode<double> displacement;
    }
    class Checkerboard : MNSample2D<Vector4D>, IMNNode<Vector4D>, INamed
    {
        public string GetName() { return "Checkerboard"; }
        public Vector4D Eval(EvalContext context)
        {
            double uvalue = u.Eval(context);
            double vvalue = v.Eval(context);
            int ucheck = (int)((uvalue - Math.Floor(uvalue)) * 2);
            int vcheck = (int)((vvalue - Math.Floor(vvalue)) * 2);
            return ((ucheck + vcheck) & 1) == 0 ? Color1.Eval(context) : Color2.Eval(context);
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double));
            var tempv = Expression.Parameter(typeof(double));
            var checku = Expression.Parameter(typeof(int));
            var checkv = Expression.Parameter(typeof(int));
            return Expression.Block(
                typeof(Vector4D),
                new ParameterExpression[] { tempu, tempv, checku, checkv },
                new Expression[]
                {
                    Expression.Assign(tempu, u.CreateExpression(evalcontext)),
                    Expression.Assign(tempv, v.CreateExpression(evalcontext)),
                    Expression.Assign(checku, Expression.Convert(Expression.Multiply(Expression.Subtract(tempu, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempu })), Expression.Constant(2.0)), typeof(int))),
                    Expression.Assign(checkv, Expression.Convert(Expression.Multiply(Expression.Subtract(tempv, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv })), Expression.Constant(2.0)), typeof(int))),
                    Expression.Condition(
                        Expression.Equal(Expression.And(Expression.Add(checku, checkv), Expression.Constant(1)), Expression.Constant(0)),
                        color1.CreateExpression(evalcontext),
                        color2.CreateExpression(evalcontext)),
                });
        }
        public IMNNode<Vector4D> Color1 { get { return color1; } set { color1 = value; } }
        public IMNNode<Vector4D> Color2 { get { return color2; } set { color2 = value; } }
        protected IMNNode<Vector4D> color1, color2;
    }
    class Perlin2D : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string GetName() { return "Perlin (2D)"; }
        static double Interpolate(double a, double b, double x)
        {
            return a * (1 - x) + b * x;
        }
        static double Noise1(int x, int y)
        {
            int n = x + y * 57;
            n = (n << 13) ^ n;
            return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        }
        static double Noise(double x, double y)
        {
            return Noise1((int)x, (int)y);
        }
        static double SmoothNoise(double x, double y)
        {
            double corners = (Noise(x - 1, y - 1) + Noise(x + 1, y - 1) + Noise(x - 1, y + 1) + Noise(x + 1, y + 1)) / 16;
            double sides = (Noise(x - 1, y) + Noise(x + 1, y) + Noise(x, y - 1) + Noise(x, y + 1)) / 8;
            double center = Noise(x, y) / 4;
            return corners + sides + center;
        }
        static double InterpolatedNoise(double x, double y)
        {
            int ix = (int)x;
            double fx = x - ix;
            int iy = (int)y;
            double fy = y - iy;
            return Interpolate(
                Interpolate(SmoothNoise(ix, iy), SmoothNoise(ix + 1, iy), fx),
                Interpolate(SmoothNoise(ix, iy + 1), SmoothNoise(ix + 1, iy + 1), fx),
                fy);
        }
        public static double PerlinNoise2D(double x, double y)
        {
            double sum = 0;
            for (int i = 0; i < 4; ++i)
            {
                double frequency = Math.Pow(2, i);
                double amplitude = Math.Pow(0.5, i);
                sum = sum + InterpolatedNoise(x * frequency, y * frequency) * amplitude;
            }
            return sum;
        }
        public Vector4D SampleTexture(double u, double v)
        {
            double p = PerlinNoise2D(u, v);
            return new Vector4D(p, p, p, 1);
        }
        public double Eval(EvalContext context) { return PerlinNoise2D(u.Eval(context), v.Eval(context)); }
        // TODO: Fill this in.
        public Expression CreateExpression(Expression evalcontext)
        {
            return System.Linq.Expressions.Expression.Call(
                Expression.Constant(this),
                typeof(Perlin2D).GetMethod("Eval"),
                new Expression[] { evalcontext });
        }
    }
    public class GenericMaterial : IMNNode<Vector4D>, INamed
    {
        public GenericMaterial(string name, Vector4D ambient, Vector4D diffuse, Vector4D specular, Vector4D reflect, Vector4D refract, double ior)
        {
            Name = name;
            Ambient = ambient;
            Diffuse = diffuse;
            Specular = specular;
            Reflect = reflect;
            Refract = refract;
            Ior = ior;
        }
        public string GetName()
        {
            return Name;
        }
        public bool IsConstant()
        {
            return true;
        }
        public Vector4D Eval(EvalContext context)
        {
            return Diffuse;
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Constant(Diffuse);
        }
        public readonly string Name;
        public readonly Vector4D Ambient;
        public readonly Vector4D Diffuse;
        public readonly Vector4D Specular;
        public readonly Vector4D Reflect;
        public readonly Vector4D Refract;
        public readonly double Ior;
    }
}