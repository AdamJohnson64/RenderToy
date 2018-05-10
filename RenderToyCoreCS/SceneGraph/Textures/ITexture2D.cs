using RenderToy.MaterialNetwork;
using System;

namespace RenderToy.Textures
{
    interface ITexture2D
    {
        Vector4D SampleTexture(double u, double v);
    }
    class TextureBrick : ITexture2D
    {
        public static double BrickMask(double u, double v)
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
        public static double BrickNoise(double u, double v)
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
            return TexturePerlin.PerlinNoise2D(u * 8, v * 8);
        }
        public Vector4D SampleTexture(double u, double v)
        {
            u = u * 4;
            v = v * 4;
            // Calculate Perlin sources.
            double perlinlow = TexturePerlin.PerlinNoise2D(u * 16, v * 16) * 0.1;
            double perlinmid = TexturePerlin.PerlinNoise2D(u * 64, v * 64);
            perlinmid = MathHelp.Saturate(perlinmid) * 1.25;
            double perlinhigh = TexturePerlin.PerlinNoise2D(u * 512, v * 512) * 0.1;
            double perlinband = TexturePerlin.PerlinNoise2D(u * 64, v * 512) * 0.2;
            // Calculate brick and mortar colors.
            Vector4D BrickColor = new Vector4D(0.5 + perlinband + perlinlow + BrickNoise(u, v) * 0.1, 0.0, 0.0, 1);
            Vector4D MortarColor = new Vector4D(0.4 + perlinhigh + perlinlow, 0.4 + perlinhigh + perlinlow, 0.4 + perlinhigh + perlinlow, 1);
            // Calculate brickness mask.
            double brickness = BrickMask(u, v) - perlinmid;
            return brickness < 0.5 ? MortarColor : BrickColor;
        }
        public static IMNNode<Vector4D> Create()
        {
            var val0 = new IMConstant<double> { Value = 0.0 };
            var val05 = new IMConstant<double> { Value = 0.5 };
            var val1 = new IMConstant<double> { Value = 1.0 };
            var val16 = new IMConstant<double> { Value = 16.0 };
            var val64 = new IMConstant<double> { Value = 64.0 };
            var val512 = new IMConstant<double> { Value = 512.0 };
            var texu = new IMMultiply { Lhs = new IMTexCoordU(), Rhs = new IMConstant<double> { Value = 4.0 } };
            var texv = new IMMultiply { Lhs = new IMTexCoordV(), Rhs = new IMConstant<double> { Value = 4.0 } };
            var perlinlow = new IMPerlin2D { U = new IMMultiply { Lhs = texu, Rhs = val16 }, V = new IMMultiply { Lhs = texv, Rhs = val16 } };
            var perlinmid = new IMPerlin2D { U = new IMMultiply { Lhs = texu, Rhs = val64 }, V = new IMMultiply { Lhs = texv, Rhs = val64 } };
            var perlinhigh = new IMPerlin2D { U = new IMMultiply { Lhs = texu, Rhs = val512 }, V = new IMMultiply { Lhs = texv, Rhs = val512 } };
            var perlinband = new IMPerlin2D { U = new IMMultiply { Lhs = texu, Rhs = val64 }, V = new IMMultiply { Lhs = texv, Rhs = val512 } };
            var perlinlowscale = new IMMultiply { Lhs = perlinlow, Rhs = new IMConstant<double> { Value = 0.1 } };
            var perlinmidscale = new IMMultiply { Lhs = new IMSaturate { Value = perlinmid }, Rhs = new IMConstant<double> { Value = 1.25 } };
            var perlinhighscale = new IMMultiply { Lhs = perlinhigh, Rhs = new IMConstant<double> { Value = 0.1 } };
            var perlinbandscale = new IMMultiply { Lhs = perlinband, Rhs = new IMConstant<double> { Value = 0.2 } };
            var brickmask = new IMThreshold { Value = new IMSubtract { Lhs = new IMBrickMask { U = texu, V = texv }, Rhs = perlinmidscale } };
            var bricknoise = new IMMultiply { Lhs = new IMBrickNoise { U = texu, V = texv }, Rhs = new IMConstant<double> { Value = 0.1 } };
            var brickcolor = new IMAdd { Lhs = new IMAdd { Lhs = val05, Rhs = perlinbandscale }, Rhs = bricknoise };
            var mortarcolor = new IMAdd { Lhs = new IMConstant<double> { Value = 0.4 }, Rhs = new IMAdd { Lhs = perlinhighscale, Rhs = perlinlowscale } };
            var colorr = new IMLerp { Value0 = mortarcolor, Value1 = brickcolor, Factor = brickmask };
            var colorg = new IMLerp { Value0 = mortarcolor, Value1 = val0, Factor = brickmask };
            var colorb = new IMLerp { Value0 = mortarcolor, Value1 = val0, Factor = brickmask };
            return new IMVector4D { R = colorr, G = colorg, B = colorb, A = val1 };
        }
    }
    class TexturePerlin : ITexture2D
    {
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
            double v1 = SmoothNoise(ix, iy);
            double v2 = SmoothNoise(ix + 1, iy);
            double v3 = SmoothNoise(ix, iy + 1);
            double v4 = SmoothNoise(ix + 1, iy + 1);
            double i1 = Interpolate(v1, v2, fx);
            double i2 = Interpolate(v3, v4, fx);
            return Interpolate(i1, i2, fy);
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
    }
    class TextureNetwork : ITexture2D
    {
        public Vector4D SampleTexture(double u, double v)
        {
            EvalContext context = new EvalContext();
            context.U = u;
            context.V = v;
            return network.Eval(context);
        }
        static IMNNode<Vector4D> network = TextureBrick.Create();
    }
}