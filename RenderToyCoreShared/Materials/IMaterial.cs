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
    public class ExpressionBase
    {
        static Expression<Func<double, double, double, double>> LerpFn2 = (value0, value1, factor) => value0 * (1 - factor) + value1 * factor;
        public static Expression<Func<double, double, double, double>> LerpFn = ExpressionReducer.Reduce(LerpFn2);
        public static Func<double, double, double, double> Lerp = LerpFn.Compile();
        public static Expression<Func<double, double, double>> PowFn = (mantissa, exponent) => Math.Pow(mantissa, exponent);
        public static Func<double, double, double> Pow = PowFn.Compile();
        static Expression<Func<double, double>> SaturateFn2 = (f) => f < 0 ? 0 : (f < 1 ? f : 1);
        public static Expression<Func<double, double>> SaturateFn = ExpressionReducer.Reduce(SaturateFn2);
        public static Func<double, double> Saturate = SaturateFn.Compile();
        static Expression<Func<double, double>> TileFn2 = (f) => f - Math.Floor(f);
        public static Expression<Func<double, double>> TileFn = ExpressionReducer.Reduce(TileFn2);
        public static Func<double, double> Tile = TileFn.Compile();
        public static Expression InvokeLerp(Expression value0, Expression value1, Expression factor)
        {
            return Expression.Invoke(LerpFn, new Expression[] { value0, value1, factor });
        }
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
    abstract class MNUnary<T> : ExpressionBase
    {
        public bool IsConstant() { return value.IsConstant(); }
        public IMNNode<T> Value { get { return value; } set { this.value = value; } }
        protected IMNNode<T> value;
    }
    abstract class MNBinary<T> : ExpressionBase
    {
        public bool IsConstant() { return lhs.IsConstant() && rhs.IsConstant(); }
        public IMNNode<T> Lhs { get { return lhs; } set { lhs = value; } }
        public IMNNode<T> Rhs { get { return rhs; } set { rhs = value; } }
        protected IMNNode<T> lhs, rhs;
    }
    abstract class MNSample2D<T> : ExpressionBase
    {
        public bool IsConstant() { return u.IsConstant() && v.IsConstant(); }
        public IMNNode<double> U { get { return u; } set { u = value; } }
        public IMNNode<double> V { get { return v; } set { v = value; } }
        protected IMNNode<double> u, v;
    }
    class MNTexCoordU : ExpressionBase, IMNNode<double>, INamed
    {
        public string Name { get { return "U"; } }
        public bool IsConstant() { return false; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Field(evalcontext, evalcontext.Type.GetField("U"));
        }
    }
    sealed class MNTexCoordV : IMNNode<double>, INamed
    {
        public string Name { get { return "V"; } }
        public bool IsConstant() { return false; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Field(evalcontext, evalcontext.Type.GetField("V"));
        }
    }
    sealed class MNConstant : IMNNode<double>, INamed
    {
        public string Name { get { return value.ToString(); } }
        public bool IsConstant() { return true; }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Constant(value);
        }
        public double Value { get { return value; } set { this.value = value; } }
        protected double value;
    }
    sealed class MNVector4D : IMNNode<Vector4D>, INamed
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
    sealed class MNAdd : MNBinary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "+"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Add(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
    sealed class MNSubtract : MNBinary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "-"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Subtract(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
    sealed class MNMultiply : MNBinary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "X"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Multiply(Lhs.CreateExpression(evalcontext), Rhs.CreateExpression(evalcontext));
        }
    }
    sealed class MNPower : ExpressionBase, IMNNode<double>, INamed
    {
        public string Name { get { return "Power"; } }
        public bool IsConstant() { return value.IsConstant() && exponent.IsConstant(); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return Expression.Invoke(PowFn, Value.CreateExpression(evalcontext), Exponent.CreateExpression(evalcontext));
        }
        public IMNNode<double> Value { get { return this.value; } set { this.value = value; } }
        public IMNNode<double> Exponent { get { return exponent; } set { exponent = value; } }
        protected IMNNode<double> value, exponent;
    }
    sealed class MNSaturate : MNUnary<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Saturate"; } }
        public static Expression CreateSaturate(Expression v)
        {
            var temp = Expression.Parameter(typeof(double), "Temp");
            return Expression.Block(typeof(double), new ParameterExpression[] { temp }, new Expression[] {
                Expression.Assign(temp, v),
                Expression.Invoke(SaturateFn, temp)
            });
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            return CreateSaturate(value.CreateExpression(evalcontext));
        }
    }
    sealed class MNSin : MNUnary<double>, IMNNode<double>, INamed
    {
        public static Expression<Func<double, double>> Sin = (f) => Math.Sin(f);
        public string Name { get { return "Sin"; } }
        public Expression CreateExpression(Expression evalcontext) { return Expression.Invoke(Sin, Value.CreateExpression(evalcontext)); }
    }
    sealed class MNThreshold : MNUnary<double>, IMNNode<double>, INamed
    {
        public static Expression<Func<double, double>> Threshold = (f) => f < 0.5 ? 0 : 1;
        public string Name { get { return "Threshold"; } }
        public Expression CreateExpression(Expression evalcontext) { return Expression.Invoke(Threshold, Value.CreateExpression(evalcontext)); }
    }
    sealed class MNLerp : ExpressionBase, IMNNode<double>, INamed
    {
        public string Name { get { return "Lerp"; } }
        public bool IsConstant() { return value0.IsConstant() && value1.IsConstant() && factor.IsConstant(); }
        public Expression CreateExpression(Expression evalcontext)
        {
            return InvokeLerp(value0.CreateExpression(evalcontext), value1.CreateExpression(evalcontext), factor.CreateExpression(evalcontext));
        }
        public IMNNode<double> Value0 { get { return value0; } set { value0 = value; } }
        public IMNNode<double> Value1 { get { return value1; } set { value1 = value; } }
        public IMNNode<double> Factor { get { return factor; } set { factor = value; } }
        protected IMNNode<double> value0, value1, factor;
    }
    sealed class Spike : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Spike"; } }
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
                    InvokeLerp(
                        Expression.Constant(1.0),
                        Expression.Constant(0.0),
                        MNSaturate.CreateSaturate(
                            Expression.Multiply(
                                Expression.Call(null, typeof(Math).GetMethod("Sqrt"), new Expression[]
                                {
                                    Expression.Add(Expression.Multiply(tempu, tempu), Expression.Multiply(tempv, tempv))
                                }),
                                Expression.Constant(2.0))))
                });
        }
    }
    sealed class BrickMask : MNSample2D<double>, IMNNode<double>, INamed
    {
        const double MortarWidth = 0.025;
        public static Expression<Func<double, double, double>> Temp = (u, v) => (v < MortarWidth) ? 0 : (((v < 0.5 - MortarWidth) ? ((u < MortarWidth) ? 0 : ((u < 1.0 - MortarWidth) ? 1 : 0)) : (v < 0.5 + MortarWidth) ? 0 : ((v < 1.0 - MortarWidth) ? (u < 0.5 - MortarWidth) ? 1 : ((u < 0.5 + MortarWidth) ? 0 : 1) : 0)));
        public static Func<double, double, double> CallTemp = Temp.Compile();
        public static double Compute(double u, double v)
        {
            return CallTemp(u - Math.Floor(u), v - Math.Floor(v)); 
        }
        public string Name { get { return "Brick Mask"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            const double MortarWidth = 0.025;
            var tempu = Expression.Parameter(typeof(double), "SampleU");
            var tempv = Expression.Parameter(typeof(double), "SampleV");
            var tileu = Expression.Parameter(typeof(double), "TiledU");
            var tilev = Expression.Parameter(typeof(double), "TiledV");
            return Expression.Block(typeof(double),
                new ParameterExpression[] { tempu, tempv, tileu, tilev },
                new Expression[]
                {
                    Expression.Assign(tempu, u.CreateExpression(evalcontext)),
                    Expression.Assign(tempv, v.CreateExpression(evalcontext)),
                    Expression.Assign(tileu, Expression.Invoke(TileFn, tempu)),
                    Expression.Assign(tilev, Expression.Invoke(TileFn, tempv)),
                    Expression.Invoke(Temp, tileu, tilev)
                });
        }
    }
    sealed class BrickNoise : MNSample2D<double>, IMNNode<double>, INamed
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
        public string Name { get { return "Brick Noise"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double), "SampleU");
            var tempv = Expression.Parameter(typeof(double), "SampleV");
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
    sealed class BumpGenerate : MNSample2D<Vector4D>, IMNNode<Vector4D>, INamed
    {
        public string Name { get { return "Bump Generate"; } }
        public bool IsConstant() { return displacement.IsConstant(); }
        static Expression _ReconstructSampler2(ParameterExpression ec, ParameterExpression newu, ParameterExpression newv)
        {
            var exprec = Expression.Variable(typeof(EvalContext));
            return Expression.Block(typeof(EvalContext),
                new ParameterExpression[] { exprec },
                new Expression[]
                {
                    Expression.Assign(exprec, Expression.New(typeof(EvalContext).GetConstructor(new Type[] { typeof(EvalContext) }), ec)),
                    Expression.Assign(Expression.Field(exprec, "U"), newu),
                    Expression.Assign(Expression.Field(exprec, "V"), newv),
                    exprec
                }
            );
        }
        static LambdaExpression _ReconstructSampler()
        {
            var ec = Expression.Parameter(typeof(EvalContext));
            var du = Expression.Parameter(typeof(double));
            var dv = Expression.Parameter(typeof(double));
            return Expression.Lambda(_ReconstructSampler2(ec, du, dv), new ParameterExpression[] { ec, du, dv});
        }
        static LambdaExpression ReconstructSampler = _ReconstructSampler();
        public Expression CreateExpression(Expression evalcontext)
        {
            var u = Expression.Parameter(typeof(double), "SampleU");
            var v = Expression.Parameter(typeof(double), "SampleV");
            var du1 = Expression.Parameter(typeof(double), "NegU");
            var du2 = Expression.Parameter(typeof(double), "PosU");
            var dv1 = Expression.Parameter(typeof(double), "NegV");
            var dv2 = Expression.Parameter(typeof(double), "PosV");
            var normal = Expression.Parameter(typeof(Vector3D), "Normal");
            return Expression.Block(typeof(Vector4D), new ParameterExpression[] { u, v, du1, du2, dv1, dv2, normal },
                new Expression[]
                {
                    Expression.Assign(u, U.CreateExpression(evalcontext)),
                    Expression.Assign(v, V.CreateExpression(evalcontext)),
                    Expression.Assign(du1, Displacement.CreateExpression(Expression.Invoke(ReconstructSampler, evalcontext, Expression.Subtract(u, Expression.Constant(0.001)), v))),
                    Expression.Assign(du2, Displacement.CreateExpression(Expression.Invoke(ReconstructSampler, evalcontext, Expression.Add(u, Expression.Constant(0.001)), v))),
                    Expression.Assign(dv1, Displacement.CreateExpression(Expression.Invoke(ReconstructSampler, evalcontext, u, Expression.Subtract(v, Expression.Constant(0.001))))),
                    Expression.Assign(dv2, Displacement.CreateExpression(Expression.Invoke(ReconstructSampler, evalcontext, u, Expression.Add(v, Expression.Constant(0.001))))),
                    Expression.Assign(normal, Expression.Call(null, typeof(MathHelp).GetMethod("Normalized", new Type[] { typeof(Vector3D) }),
                        Expression.New(typeof(Vector3D).GetConstructor(new Type[] { typeof(double), typeof(double), typeof(double) }),
                        Expression.Divide(Expression.Subtract(du1, du2), Expression.Constant(0.002)),
                        Expression.Divide(Expression.Subtract(dv1, dv2), Expression.Constant(0.002)),
                        Expression.Constant(1.0)
                    ))),
                    Expression.New(typeof(Vector4D).GetConstructor(new Type[] { typeof(double), typeof(double), typeof(double), typeof(double) }),
                        Expression.Add(Expression.Multiply(Expression.Field(normal, "X"), Expression.Constant(0.5)), Expression.Constant(0.5)),
                        Expression.Add(Expression.Multiply(Expression.Field(normal, "Y"), Expression.Constant(0.5)), Expression.Constant(0.5)),
                        Expression.Add(Expression.Multiply(Expression.Field(normal, "Z"), Expression.Constant(0.5)), Expression.Constant(0.5)),
                        Expression.Constant(1.0))
                });
        }
        public IMNNode<double> Displacement
        {
            get { return displacement; }
            set { displacement = value; }
        }
        IMNNode<double> displacement;
    }
    sealed class Checkerboard : MNSample2D<Vector4D>, IMNNode<Vector4D>, INamed
    {
        public string Name { get { return "Checkerboard"; } }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double), "SampleU");
            var tempv = Expression.Parameter(typeof(double), "SampleV");
            var intu = Expression.Parameter(typeof(int), "TiledU");
            var intv = Expression.Parameter(typeof(int), "TiledV");
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
    sealed class Perlin2D : MNSample2D<double>, IMNNode<double>, INamed
    {
        public string Name { get { return "Perlin (2D)"; } }
        public static Expression<Func<int, int, int>> Temp1 = (x, y) => x + y * 57;
        public static Expression<Func<int, int>> Temp2 = (n) => (n << 13) ^ n;
        public static Expression<Func<int, double>> Temp3 = (n) => 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        public static double Random2D(int x, int y)
        {
            int n = x + y * 57;
            n = (n << 13) ^ n;
            return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        }
        public static Expression Random2D(Expression x, Expression y)
        {
            var _temp = Expression.Invoke(Temp1, x, y);
            var _temp2 = Expression.Invoke(Temp2, _temp);
            return Expression.Invoke(Temp3, _temp2);
        }
        public static double Noise2D(double x, double y)
        {
            return Random2D((int)x, (int)y);
        }
        public static Expression _Noise2D(Expression x, Expression y)
        {
            return Random2D(Expression.Convert(x, typeof(int)), Expression.Convert(y, typeof(int)));
        }
        public static Expression _Noise2D()
        {
            var xp = Expression.Parameter(typeof(double));
            var yp = Expression.Parameter(typeof(double));
            return Expression.Lambda(_Noise2D(xp, yp), "Noise2D", new[] { xp, yp });
        }
        public static Expression Noise2DFn = _Noise2D();
        public static Expression Noise2D(Expression x, Expression y)
        {
            return Expression.Invoke(Noise2DFn, x, y);
        }
        public static double SmoothNoise(double x, double y)
        {
            double corners = (Noise2D(x - 1, y - 1) + Noise2D(x + 1, y - 1) + Noise2D(x - 1, y + 1) + Noise2D(x + 1, y + 1)) / 16;
            double sides = (Noise2D(x - 1, y) + Noise2D(x + 1, y) + Noise2D(x, y - 1) + Noise2D(x, y + 1)) / 8;
            double center = Noise2D(x, y) / 4;
            return corners + sides + center;
        }
        public static Expression _SmoothNoise(Expression x, Expression y)
        {
            var xminus1 = Expression.Subtract(x, Expression.Constant(1.0));
            var xplus1 = Expression.Add(x, Expression.Constant(1.0));
            var yminus1 = Expression.Subtract(y, Expression.Constant(1.0));
            var yplus1 = Expression.Add(y, Expression.Constant(1.0));
            var corners =
                Expression.Divide(
                    Expression.Add(Noise2D(xminus1, yminus1), Expression.Add(Noise2D(xplus1, yminus1), Expression.Add(Noise2D(xminus1, yplus1), Noise2D(xplus1, yplus1)))),
                    Expression.Constant(16.0));
            var sides =
                Expression.Divide(
                    Expression.Add(Noise2D(xminus1, y), Expression.Add(Noise2D(xplus1, y), Expression.Add(Noise2D(x, yminus1), Noise2D(x, yplus1)))),
                    Expression.Constant(8.0));
            var center = Expression.Divide(Noise2D(x, y), Expression.Constant(4.0));
            return Expression.Add(corners, Expression.Add(sides, center));
        }
        public static Expression _SmoothNoise()
        {
            var xp = Expression.Parameter(typeof(double));
            var yp = Expression.Parameter(typeof(double));
            return Expression.Lambda(_SmoothNoise(xp, yp), "SmoothNoise", new[] { xp, yp });
        }
        public static Expression SmoothNoiseFn = _SmoothNoise();
        public static Expression SmoothNoise(Expression x, Expression y)
        {
            return Expression.Invoke(SmoothNoiseFn, x, y);
        }
        public static double InterpolatedNoise(double x, double y)
        {
            int ix = (int)x;
            double fx = x - ix;
            int iy = (int)y;
            double fy = y - iy;
            return Lerp(
                Lerp(SmoothNoise(ix, iy), SmoothNoise(ix + 1, iy), fx),
                Lerp(SmoothNoise(ix, iy + 1), SmoothNoise(ix + 1, iy + 1), fx),
                fy);
        }
        public static Expression _InterpolatedNoise(Expression x, Expression y)
        {
            var tempix = Expression.Parameter(typeof(double), "SampleURounded");
            var tempiy = Expression.Parameter(typeof(double), "SampleVRounded");
            var tempfx = Expression.Parameter(typeof(double), "TiledU");
            var tempfy = Expression.Parameter(typeof(double), "TiledV");
            return Expression.Block(
                new ParameterExpression[] { tempix, tempiy, tempfx, tempfy },
                new Expression[]
                {
                    Expression.Assign(tempix, Expression.Convert(Expression.Convert(x, typeof(int)), typeof(double))),
                    Expression.Assign(tempiy, Expression.Convert(Expression.Convert(y, typeof(int)), typeof(double))),
                    Expression.Assign(tempfx, Expression.Subtract(x, tempix)),
                    Expression.Assign(tempfy, Expression.Subtract(y, tempiy)),
                    InvokeLerp(
                        InvokeLerp(SmoothNoise(tempix, tempiy), SmoothNoise(Expression.Add(tempix, Expression.Constant(1.0)), tempiy), tempfx),
                        InvokeLerp(SmoothNoise(tempix, Expression.Add(tempiy, Expression.Constant(1.0))), SmoothNoise(Expression.Add(tempix, Expression.Constant(1.0)), Expression.Add(tempiy, Expression.Constant(1.0))), tempfx),
                        tempfy)
                });
        }
        public static Expression _InterpolatedNoise()
        {
            var x = Expression.Parameter(typeof(double));
            var y = Expression.Parameter(typeof(double));
            return Expression.Lambda(_InterpolatedNoise(x, y), "InterpolatedNoise", new[] { x, y });
        }
        public static Expression InterpolatedNoiseFn = _InterpolatedNoise();
        public static Expression InterpolatedNoise(Expression x, Expression y)
        {
            return Expression.Invoke(InterpolatedNoiseFn, x, y);
        }
        public static double PerlinNoise2D(double x, double y)
        {
            return
                InterpolatedNoise(x * 1.0, y * 1.0) * 1.0 +
                InterpolatedNoise(x * 2.0, y * 2.0) * 0.5 +
                InterpolatedNoise(x * 4.0, y * 4.0) * 0.25 +
                InterpolatedNoise(x * 8.0, y * 8.0) * 0.125;
        }
        public static Expression _PerlinNoise2D(Expression x, Expression y)
        {
            return Expression.Add(
                Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(1.0)), Expression.Multiply(y, Expression.Constant(1.0))), Expression.Constant(1.0)),
                Expression.Add(
                    Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(2.0)), Expression.Multiply(y, Expression.Constant(2.0))), Expression.Constant(0.5)),
                    Expression.Add(
                        Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(4.0)), Expression.Multiply(y, Expression.Constant(4.0))), Expression.Constant(0.25)),
                        Expression.Multiply(InterpolatedNoise(Expression.Multiply(x, Expression.Constant(8.0)), Expression.Multiply(y, Expression.Constant(8.0))), Expression.Constant(0.125)))));
        }
        public static Expression _PerlinNoise2D()
        {
            var x = Expression.Parameter(typeof(double));
            var y = Expression.Parameter(typeof(double));
            return Expression.Lambda(_PerlinNoise2D(x, y), "PerlinNoise2D", new[] { x, y });
        }
        public static Expression PerlinNoiseFn = _PerlinNoise2D();
        public static Expression PerlinNoise2D(Expression x, Expression y)
        {
            return Expression.Invoke(PerlinNoiseFn, x, y);
        }
        public Vector4D SampleTexture(double u, double v)
        {
            double p = PerlinNoise2D(u, v);
            return new Vector4D(p, p, p, 1);
        }
        public Expression CreateExpression(Expression evalcontext)
        {
            var tempu = Expression.Parameter(typeof(double), "SampleU");
            var tempv = Expression.Parameter(typeof(double), "SampleV");
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