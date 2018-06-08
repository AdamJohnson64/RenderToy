////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////


using RenderToy.Math;

namespace RenderToy.Materials
{
    public static class StockMaterials
    {
        public static Vector4D Empty = new Vector4D(0, 0, 0, 0);
        public static Vector4D Percent0 = new Vector4D(0, 0, 0, 0);
        public static Vector4D Percent50 = new Vector4D(0.5, 0.5, 0.5, 0.5);
        public static Vector4D Percent100 = new Vector4D(1, 1, 1, 1);
        public static Vector4D Black = new Vector4D(0, 0, 0, 1);
        public static Vector4D Red = new Vector4D(1, 0, 0, 1);
        public static Vector4D Green = new Vector4D(0, 0.5, 0, 1);
        public static Vector4D Blue = new Vector4D(0, 0, 1, 1);
        public static Vector4D Yellow = new Vector4D(1, 1, 0, 1);
        public static Vector4D Magenta = new Vector4D(1, 0, 1, 1);
        public static Vector4D Cyan = new Vector4D(0, 1, 1, 1);
        public static Vector4D White = new Vector4D(1, 1, 1, 1);
        public static Vector4D DarkGray = new Vector4D(0.25, 0.25, 0.25, 1);
        public static Vector4D LightGray = new Vector4D(0.75, 0.75, 0.75, 1);
        public static Vector4D LightBlue = new Vector4D(0.5, 0.5, 1.0, 1);
        public static GenericMaterial PlasticBlack = new GenericMaterial("Black Plastic", Empty, Black, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticRed = new GenericMaterial("Red Plastic", Empty, Red, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticGreen = new GenericMaterial("Green Plastic", Empty, Green, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticBlue = new GenericMaterial("Blue Plastic", Empty, Blue, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticYellow = new GenericMaterial("Yellow Plastic", Empty, Yellow, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticMagenta = new GenericMaterial("Magenta Plastic", Empty, Magenta, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticCyan = new GenericMaterial("Cyan Plastic", Empty, Cyan, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticWhite = new GenericMaterial("White Plastic", Empty, White, White, Percent50, Percent0, 1);
        public static GenericMaterial Glass = new GenericMaterial("Glass", Empty, Empty, White, Percent0, Percent100, 1.5);
        public static GenericMaterial PlasticLightBlue = new GenericMaterial("Light Blue Plastic", Empty, LightBlue, White, Percent50, Percent0, 1);
        public static IMNNode<Vector4D> Missing = GenerateMissing();
        public static IMNNode<Vector4D> Brick = GenerateBrick();
        public static IMNNode<Vector4D> MarbleBlack = GenerateMarbleBlack();
        public static IMNNode<Vector4D> MarbleWhite = GenerateMarbleWhite();
        public static IMNNode<Vector4D> MarbleTile = GenerateMarbleTile();
        static IMNNode<Vector4D> GenerateMissing()
        {
            return new Checkerboard { U = Multiply(TexU(), Constant(8)), V = Multiply(TexV(), Constant(8)), Color1 = PlasticMagenta, Color2 = PlasticBlack };
        }
        static IMNNode<Vector4D> GenerateBrick()
        {
            var texu = Multiply(TexU(), Constant(4));
            var texv = Multiply(TexV(), Constant(4));
            var perlinlow = new Perlin2D { U = Multiply(texu, Constant(16)), V = Multiply(texv, Constant(16)) };
            var perlinmid = new Perlin2D { U = Multiply(texu, Constant(64)), V = Multiply(texv, Constant(64)) };
            var perlinhigh = new Perlin2D { U = Multiply(texu, Constant(512)), V = Multiply(texv, Constant(512)) };
            var perlinband = new Perlin2D { U = Multiply(texu, Constant(64)), V = Multiply(texv, Constant(512)) };
            var perlinlowscale = Multiply(perlinlow, Constant(0.1));
            var perlinmidscale = Multiply(new MNSaturate { Value = perlinmid }, Constant(1.25));
            var perlinhighscale = Multiply(perlinhigh, Constant(0.1));
            var perlinbandscale = Multiply(perlinband, Constant(0.2));
            var brickmask = new MNThreshold { Value = Subtract(new BrickMask { U = texu, V = texv }, perlinmidscale) };
            var bricknoise = Multiply(new BrickNoise { U = texu, V = texv }, Constant(0.1));
            var brickcolor = Add(Add(Constant(0.5), perlinbandscale), bricknoise);
            var mortarcolor = Add(Constant(0.4), Add(perlinhighscale, perlinlowscale));
            var colorr = Lerp(mortarcolor, brickcolor, brickmask);
            var color = Lerp(mortarcolor, Constant(0), brickmask);
            return RGBA(colorr, color, color, Constant(1));
        }
        static IMNNode<Vector4D> GenerateMarbleBlack()
        {
            var perlinlow = new Perlin2D { U = Multiply(TexU(), Constant(5)), V = Multiply(TexV(), Constant(5)) };
            var perlinhigh = new Perlin2D { U = Multiply(TexU(), Constant(50)), V = Multiply(TexV(), Constant(50)) };
            var v1 = Multiply(Power(Multiply(Constant(0.5), Add(Constant(1), Sin(Multiply(Add(TexU(), Add(Multiply(perlinlow, Constant(0.5)), Multiply(perlinhigh, Constant(0.2)))), Constant(50))))), Constant(20)), Constant(0.8));
            var v2 = Multiply(Power(Multiply(Constant(0.5), Add(Constant(1), Sin(Multiply(Add(TexU(), Add(Multiply(perlinlow, Constant(0.5)), Constant(50))), Constant(100))))), Constant(20)), Constant(0.2));
            var c = Add(v1, v2);
            return RGBA(c, c, c, Constant(1));
        }
        static IMNNode<Vector4D> GenerateMarbleWhite()
        {
            var perlinlow = new Perlin2D { U = Multiply(TexU(), Constant(5)), V = Multiply(TexV(), Constant(5)) };
            var perlinhigh = new Perlin2D { U = Multiply(TexU(), Constant(50)), V = Multiply(TexV(), Constant(50)) };
            var v1 = Multiply(Power(Multiply(Constant(0.5), Add(Constant(1), Sin(Multiply(Add(TexU(), Add(Multiply(perlinlow, Constant(0.5)), Multiply(perlinhigh, Constant(0.2)))), Constant(50))))), Constant(20)), Constant(0.8));
            var v2 = Multiply(Power(Multiply(Constant(0.5), Add(Constant(1), Sin(Multiply(Add(TexU(), Add(Multiply(perlinlow, Constant(0.5)), Constant(50))), Constant(100))))), Constant(20)), Constant(0.2));
            var c = Subtract(Constant(1), Add(v1, v2));
            return RGBA(c, c, c, Constant(1));
        }
        static IMNNode<Vector4D> GenerateMarbleTile()
        {
            return new Checkerboard { U = TexU(), V = TexV(), Color1 = MarbleWhite, Color2 = MarbleBlack };
        }
        static IMNNode<double> Constant(double value) { return new MNConstant { Value = value }; }
        static IMNNode<double> TexU() { return new MNTexCoordU(); }
        static IMNNode<double> TexV() { return new MNTexCoordV(); }
        static IMNNode<double> Add(IMNNode<double> lhs, IMNNode<double> rhs) { return new MNAdd { Lhs = lhs, Rhs = rhs }; }
        static IMNNode<double> Multiply(IMNNode<double> lhs, IMNNode<double> rhs) { return new MNMultiply { Lhs = lhs, Rhs = rhs }; }
        static IMNNode<double> Subtract(IMNNode<double> lhs, IMNNode<double> rhs) { return new MNSubtract { Lhs = lhs, Rhs = rhs }; }
        static IMNNode<double> Lerp(IMNNode<double> value0, IMNNode<double> value1, IMNNode<double> factor) { return new MNLerp { Value0 = value0, Value1 = value1, Factor = factor }; }
        static IMNNode<double> Power(IMNNode<double> value, IMNNode<double> exponent) { return new MNPower { Value = value, Exponent = exponent }; }
        static IMNNode<double> Sin(IMNNode<double> value) { return new MNSin { Value = value }; }
        static IMNNode<Vector4D> RGBA(IMNNode<double> r, IMNNode<double> g, IMNNode<double> b, IMNNode<double> a) { return new MNVector4D { R = r, G = g, B = b, A = a }; }
    }
}