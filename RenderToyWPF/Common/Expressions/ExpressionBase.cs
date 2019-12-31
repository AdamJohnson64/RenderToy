using RenderToy.Expressions;
using System;
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
}