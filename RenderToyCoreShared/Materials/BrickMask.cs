using RenderToy.Utility;
using System;
using System.Linq.Expressions;

namespace RenderToy.Materials
{
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
}