////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
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
            return Expression.Field(evalcontext, typeof(EvalContext).GetField("U"));
        }
    }
    class MNTexCoordV : IMNNode<double>, INamed
    {
        public string GetName() { return "V"; }
        public bool IsConstant() { return false; }
        public double Eval(EvalContext context) { return context.V; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Field(evalcontext, typeof(EvalContext).GetField("V"));
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
            var parts = new IMNNode<double>[] { r, g, b, a }.Distinct().ToArray();
            Dictionary<IMNNode<double>, Expression> lookup = null;
            if (parts.Length < 4)
            {
                lookup = parts.ToDictionary(k => k, v => (Expression)Expression.Variable(typeof(double)));
            }
            else
            {
                lookup = parts.ToDictionary(k => k, v => v.CreateExpression(evalcontext));
            }
            Expression interior = Expression.New(
                typeof(Vector4D).GetConstructor(new System.Type[] { typeof(double), typeof(double), typeof(double), typeof(double) }),
                new Expression[] { lookup[r], lookup[b], lookup[b], lookup[a] });
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
        public double Eval(EvalContext context) { return Saturate(value.Eval(context)); }
        public static double Saturate(double v)
        {
            return v < 0 ? 0 : (v < 1 ? v : 1);
        }
        public static Expression Saturate(Expression v)
        {
            var temp = Expression.Parameter(typeof(double), "Saturate_TEMP");
            return Expression.Block(typeof(double), new ParameterExpression[] { temp }, new Expression[] {
                Expression.Assign(temp, v),
                Expression.Condition(
                    Expression.LessThan(temp, Expression.Constant(0.0)),
                    Expression.Constant(0.0),
                    Expression.Condition(
                        Expression.LessThan(temp, Expression.Constant(1.0)),
                        temp,
                        Expression.Constant(1.0)))
            });
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Saturate(value.CreateExpression(evalcontext));
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
            return Lerp(value0.CreateExpression(evalcontext), value1.CreateExpression(evalcontext), factor.CreateExpression(evalcontext));
        }
        public static double Lerp(double value0, double value1, double factor)
        {
            return value0 * (1 - factor) + value1 * factor;
        }
        public static Expression Lerp(Expression value0, Expression value1, Expression factor)
        {
            var temp = Expression.Parameter(typeof(double), "Lerp::Lerp::TEMP");
            return Expression.Block(typeof(double), new ParameterExpression[] { temp }, new Expression[]
            {
                Expression.Assign(temp, factor),
                Expression.Add(
                    Expression.Multiply(value0, Expression.Subtract(Expression.Constant(1.0), temp)),
                    Expression.Multiply(value1, temp)),
            });
        }
        public IMNNode<double> Value0 { get { return value0; } set { value0 = value; } }
        public IMNNode<double> Value1 { get { return value1; } set { value1 = value; } }
        public IMNNode<double> Factor { get { return factor; } set { factor = value; } }
        protected IMNNode<double> value0, value1, factor;
    }
    class Spike : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string GetName() { return "Spike"; }
        public double Eval(EvalContext context)
        {
            double u = context.U - 0.5;
            double v = context.V - 0.5;
            double d = MNSaturate.Saturate(Math.Sqrt(u * u + v * v) * 2);
            return MNLerp.Lerp(1.0, 0.0, d);
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Variable(typeof(double));
            var tempv = Expression.Variable(typeof(double));
            return Expression.Block(
                new ParameterExpression[] { tempu, tempv },
                new Expression[]
                {
                    Expression.Assign(tempu, Expression.Subtract(u.CreateExpression(evalcontext), Expression.Constant(0.5))),
                    Expression.Assign(tempv, Expression.Subtract(v.CreateExpression(evalcontext), Expression.Constant(0.5))),
                    MNLerp.Lerp(
                        Expression.Constant(1.0),
                        Expression.Constant(0.0),
                        MNSaturate.Saturate(
                            Expression.Multiply(
                                Expression.Call(null, typeof(Math).GetMethod("Sqrt"), new Expression[]
                                {
                                    Expression.Add(Expression.Multiply(tempu, tempu), Expression.Multiply(tempv, tempv))
                                }),
                                Expression.Constant(2.0))))
                });
        }
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
            var tempu = Expression.Parameter(typeof(double), "BrickMask::CreateExpression(u)");
            var tempv = Expression.Parameter(typeof(double), "BrickMask::CreateExpression(v)");
            var tileu = Expression.Parameter(typeof(double), "BrickMask::CreateExpression::U_TILED");
            var tilev = Expression.Parameter(typeof(double), "BrickMask::CreateExpression::V_TILED");
            return Expression.Block(typeof(double),
                new ParameterExpression[] { tempu, tempv, tileu, tilev },
                new Expression[]
                {
                    Expression.Assign(tempu, u.CreateExpression(evalcontext)),
                    Expression.Assign(tempv, v.CreateExpression(evalcontext)),
                    Expression.Assign(tileu, Expression.Subtract(tempu, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempu }))),
                    Expression.Assign(tilev, Expression.Subtract(tempv, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv }))),
                    Expression.Condition(
                        Expression.LessThan(tilev, Expression.Constant(MortarWidth)),
                        Expression.Constant(0.0),
                        Expression.Condition(
                            Expression.LessThan(tilev, Expression.Subtract(Expression.Constant(0.5), Expression.Constant(MortarWidth))),
                            Expression.Condition(
                                Expression.LessThan(tileu, Expression.Constant(MortarWidth)),
                                Expression.Constant(0.0),
                                Expression.Condition(
                                    Expression.LessThan(tileu, Expression.Subtract(Expression.Constant(1.0), Expression.Constant(MortarWidth))),
                                    Expression.Constant(1.0),
                                    Expression.Constant(0.0))),
                            Expression.Condition(
                                Expression.LessThan(tilev, Expression.Add(Expression.Constant(0.5), Expression.Constant(MortarWidth))),
                                Expression.Constant(0.0),
                                Expression.Condition(
                                    Expression.LessThan(tilev, Expression.Subtract(Expression.Constant(1.0), Expression.Constant(MortarWidth))),
                                    Expression.Condition(
                                        Expression.LessThan(tileu, Expression.Subtract(Expression.Constant(0.5), Expression.Constant(MortarWidth))),
                                        Expression.Constant(1.0),
                                        Expression.Condition(
                                            Expression.LessThan(tileu, Expression.Add(Expression.Constant(0.5), Expression.Constant(MortarWidth))),
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
                return Perlin2D.PerlinNoise2D(Math.Floor(u) * 8, Math.Floor(v + 0.5) * 8);
            }
            else
            {
                return Perlin2D.PerlinNoise2D(Math.Floor(u + 0.5) * 8, Math.Floor(v) * 8);
            }
        }
        public string GetName() { return "Brick Noise"; }
        public double Eval(EvalContext context) { return Compute(u.Eval(context), v.Eval(context)); }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double), "BrickNoise::CreateExpression(u)");
            var tempv = Expression.Parameter(typeof(double), "BrickNoise::CreateExpression(v)");
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
                        Perlin2D.PerlinNoise2D(
                                Expression.Multiply(Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempu }), Expression.Constant(8.0)),
                                Expression.Multiply(Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { Expression.Add(tempv, Expression.Constant(0.5)) }), Expression.Constant(8.0))),
                        Perlin2D.PerlinNoise2D(
                                Expression.Multiply(Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { Expression.Add(tempu, Expression.Constant(0.5)) }), Expression.Constant(8.0)),
                                Expression.Multiply(Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv }), Expression.Constant(8.0))))
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
            return new Vector4D(normal.X * 0.5 + 0.5, -normal.Y * 0.5 + 0.5, normal.Z * 0.5 + 0.5, 1);
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
            double tempu = u.Eval(context);
            double tempv = v.Eval(context);
            int intu = (int)((tempu - Math.Floor(tempu)) * 2);
            int intv = (int)((tempv - Math.Floor(tempv)) * 2);
            return ((intu + intv) & 1) == 0 ? Color1.Eval(context) : Color2.Eval(context);
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double), "CheckerBoard::CreateExpression(u)");
            var tempv = Expression.Parameter(typeof(double), "CheckerBoard::CreateExpression(v)");
            var intu = Expression.Parameter(typeof(int), "CheckerBoard::CreateExpression::U_TILED");
            var intv = Expression.Parameter(typeof(int), "CheckerBoard::CreateExpression::V_TILED");
            return Expression.Block(
                typeof(Vector4D),
                new ParameterExpression[] { tempu, tempv, intu, intv },
                new Expression[]
                {
                    Expression.Assign(tempu, u.CreateExpression(evalcontext)),
                    Expression.Assign(tempv, v.CreateExpression(evalcontext)),
                    Expression.Assign(intu, Expression.Convert(Expression.Multiply(Expression.Subtract(tempu, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempu })), Expression.Constant(2.0)), typeof(int))),
                    Expression.Assign(intv, Expression.Convert(Expression.Multiply(Expression.Subtract(tempv, Expression.Call(null, typeof(Math).GetMethod("Floor", new Type[] { typeof(double) }), new Expression[] { tempv })), Expression.Constant(2.0)), typeof(int))),
                    Expression.Condition(
                        Expression.Equal(Expression.And(Expression.Add(intu, intv), Expression.Constant(1)), Expression.Constant(0)),
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
        public static double Random2D(int x, int y)
        {
            int n = x + y * 57;
            n = (n << 13) ^ n;
            return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        }
        public static Expression Random2D(Expression x, Expression y)
        {
            var temp = Expression.Variable(typeof(int), "Perlin2D::Noise2D::TEMP1");
            var temp2 = Expression.Variable(typeof(int), "Perlin2D::Noise2D::TEMP2");
            var temp3 = Expression.Variable(typeof(int), "Perlin2D::Noise2D::TEMP3");
            return Expression.Block(
                new ParameterExpression[] { temp, temp2, temp3 },
                new Expression[]
                {
                    Expression.Assign(temp, Expression.Add(x, Expression.Multiply(y, Expression.Constant(57)))),
                    Expression.Assign(temp2, Expression.ExclusiveOr(Expression.LeftShift(temp, Expression.Constant(13)), temp)),
                    Expression.Assign(temp3,
                        Expression.And(
                            Expression.Add(Expression.Multiply(temp2, Expression.Add(Expression.Multiply(temp2, Expression.Multiply(temp2, Expression.Constant(15731))), Expression.Constant(789221))), Expression.Constant(1376312589)),
                            Expression.Constant(0x7fffffff))),
                    Expression.Subtract(Expression.Constant(1.0),
                        Expression.Divide(
                            Expression.Convert(temp3, typeof(double)),
                            Expression.Constant(1073741824.0)))

                });
        }
        public static double Noise2D(double x, double y)
        {
            return Random2D((int)x, (int)y);
        }
        public static Expression Noise2D(Expression x, Expression y)
        {
            return Random2D(Expression.Convert(x, typeof(int)), Expression.Convert(y, typeof(int)));
        }
        public static double SmoothNoise(double x, double y)
        {
            double corners = (Noise2D(x - 1, y - 1) + Noise2D(x + 1, y - 1) + Noise2D(x - 1, y + 1) + Noise2D(x + 1, y + 1)) / 16;
            double sides = (Noise2D(x - 1, y) + Noise2D(x + 1, y) + Noise2D(x, y - 1) + Noise2D(x, y + 1)) / 8;
            double center = Noise2D(x, y) / 4;
            return corners + sides + center;
        }
        public static Expression SmoothNoise(Expression x, Expression y)
        {
            var corners =
                Expression.Divide(
                    Expression.Add(
                        Noise2D(Expression.Subtract(x, Expression.Constant(1.0)), Expression.Subtract(y, Expression.Constant(1.0))),
                        Expression.Add(
                            Noise2D(Expression.Add(x, Expression.Constant(1.0)), Expression.Subtract(y, Expression.Constant(1.0))),
                            Expression.Add(
                                Noise2D(Expression.Subtract(x, Expression.Constant(1.0)), Expression.Add(y, Expression.Constant(1.0))),
                                Noise2D(Expression.Add(x, Expression.Constant(1.0)), Expression.Add(y, Expression.Constant(1.0)))))),
                    Expression.Constant(16.0));
            var sides =
                Expression.Divide(
                    Expression.Add(
                        Noise2D(Expression.Subtract(x, Expression.Constant(1.0)), y),
                        Expression.Add(
                            Noise2D(Expression.Add(x, Expression.Constant(1.0)), y),
                            Expression.Add(
                                Noise2D(x, Expression.Subtract(y, Expression.Constant(1.0))),
                                Noise2D(x, Expression.Add(y, Expression.Constant(1.0)))))),
                    Expression.Constant(8.0));
            var center = Expression.Divide(Noise2D(x, y), Expression.Constant(4.0));
            return Expression.Add(corners, Expression.Add(sides, center));
        }
        public static double InterpolatedNoise(double x, double y)
        {
            int ix = (int)x;
            double fx = x - ix;
            int iy = (int)y;
            double fy = y - iy;
            return MNLerp.Lerp(
                MNLerp.Lerp(SmoothNoise(ix, iy), SmoothNoise(ix + 1, iy), fx),
                MNLerp.Lerp(SmoothNoise(ix, iy + 1), SmoothNoise(ix + 1, iy + 1), fx),
                fy);
        }
        public static Expression InterpolatedNoise(Expression x, Expression y)
        {
            var tempix = Expression.Parameter(typeof(double), "Perlin2D::InterpolatedNoise::U_TILE");
            var tempiy = Expression.Parameter(typeof(double), "Perlin2D::InterpolatedNoise::V_TILE");
            var tempfx = Expression.Parameter(typeof(double), "Perlin2D::InterpolatedNoise::U_TILED");
            var tempfy = Expression.Parameter(typeof(double), "Perlin2D::InterpolatedNoise::V_TILED");
            return Expression.Block(
                new ParameterExpression[] { tempix, tempiy, tempfx, tempfy },
                new Expression[]
                {
                    Expression.Assign(tempix, Expression.Convert(Expression.Convert(x, typeof(int)), typeof(double))),
                    Expression.Assign(tempiy, Expression.Convert(Expression.Convert(y, typeof(int)), typeof(double))),
                    Expression.Assign(tempfx, Expression.Subtract(x, tempix)),
                    Expression.Assign(tempfy, Expression.Subtract(y, tempiy)),
                    MNLerp.Lerp(
                        MNLerp.Lerp(SmoothNoise(tempix, tempiy), SmoothNoise(Expression.Add(tempix, Expression.Constant(1.0)), tempiy), tempfx),
                        MNLerp.Lerp(SmoothNoise(tempix, Expression.Add(tempiy, Expression.Constant(1.0))), SmoothNoise(Expression.Add(tempix, Expression.Constant(1.0)), Expression.Add(tempiy, Expression.Constant(1.0))), tempfx),
                        tempfy)
                });
        }
        public static double PerlinNoise2D(double x, double y)
        {
            return
                InterpolatedNoise(x * 1.0, y * 1.0) * 1.0 +
                InterpolatedNoise(x * 2.0, y * 2.0) * 0.5 +
                InterpolatedNoise(x * 4.0, y * 4.0) * 0.25 +
                InterpolatedNoise(x * 8.0, y * 8.0) * 0.125;
        }
        public static Expression PerlinNoise2D(Expression x, Expression y)
        {
            return Expression.Add(
                Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(1.0)), Expression.Multiply(y, Expression.Constant(1.0))), Expression.Constant(1.0)),
                Expression.Add(
                    Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(2.0)), Expression.Multiply(y, Expression.Constant(2.0))), Expression.Constant(0.5)),
                    Expression.Add(
                        Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(4.0)), Expression.Multiply(y, Expression.Constant(4.0))), Expression.Constant(0.25)),
                        Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(8.0)), Expression.Multiply(y, Expression.Constant(8.0))), Expression.Constant(0.125)))));
        }
        public Vector4D SampleTexture(double u, double v)
        {
            double p = PerlinNoise2D(u, v);
            return new Vector4D(p, p, p, 1);
        }
        public double Eval(EvalContext context) { return PerlinNoise2D(u.Eval(context), v.Eval(context)); }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double), "Perlin2D::CreateExpression(u)");
            var tempv = Expression.Parameter(typeof(double), "Perlin2D::CreateExpression(v)");
            return Expression.Block(
                new ParameterExpression[] { tempu, tempv },
                new Expression[]
                {
                    Expression.Assign(tempu, u.CreateExpression(evalcontext)),
                    Expression.Assign(tempv, v.CreateExpression(evalcontext)),
                    PerlinNoise2D(tempu, tempv)
                });
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