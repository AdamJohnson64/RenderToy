////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Utility;

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
        public static GenericMaterial PlasticRed = new GenericMaterial(Empty, Red, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticGreen = new GenericMaterial(Empty, Green, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticBlue = new GenericMaterial(Empty, Blue, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticYellow = new GenericMaterial(Empty, Yellow, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticMagenta = new GenericMaterial(Empty, Magenta, White, Percent50, Percent0, 1);
        public static GenericMaterial PlasticCyan = new GenericMaterial(Empty, Cyan, White, Percent50, Percent0, 1);
        public static GenericMaterial Glass = new GenericMaterial(Empty, Empty, White, Percent0, Percent100, 1.5);
        public static IMNNode<Vector4D> Brick()
        {
            var val0 = new MNConstant { Value = 0.0 };
            var val05 = new MNConstant { Value = 0.5 };
            var val1 = new MNConstant { Value = 1.0 };
            var val16 = new MNConstant { Value = 16.0 };
            var val64 = new MNConstant { Value = 64.0 };
            var val512 = new MNConstant { Value = 512.0 };
            var texu = new MNMultiply { Lhs = new MNTexCoordU(), Rhs = new MNConstant { Value = 4.0 } };
            var texv = new MNMultiply { Lhs = new MNTexCoordV(), Rhs = new MNConstant { Value = 4.0 } };
            var perlinlow = new MNPerlin2D { U = new MNMultiply { Lhs = texu, Rhs = val16 }, V = new MNMultiply { Lhs = texv, Rhs = val16 } };
            var perlinmid = new MNPerlin2D { U = new MNMultiply { Lhs = texu, Rhs = val64 }, V = new MNMultiply { Lhs = texv, Rhs = val64 } };
            var perlinhigh = new MNPerlin2D { U = new MNMultiply { Lhs = texu, Rhs = val512 }, V = new MNMultiply { Lhs = texv, Rhs = val512 } };
            var perlinband = new MNPerlin2D { U = new MNMultiply { Lhs = texu, Rhs = val64 }, V = new MNMultiply { Lhs = texv, Rhs = val512 } };
            var perlinlowscale = new MNMultiply { Lhs = perlinlow, Rhs = new MNConstant { Value = 0.1 } };
            var perlinmidscale = new MNMultiply { Lhs = new MNSaturate { Value = perlinmid }, Rhs = new MNConstant { Value = 1.25 } };
            var perlinhighscale = new MNMultiply { Lhs = perlinhigh, Rhs = new MNConstant { Value = 0.1 } };
            var perlinbandscale = new MNMultiply { Lhs = perlinband, Rhs = new MNConstant { Value = 0.2 } };
            var brickmask = new MNThreshold { Value = new MNSubtract { Lhs = new MNBrickMask { U = texu, V = texv }, Rhs = perlinmidscale } };
            var bricknoise = new MNMultiply { Lhs = new MNBrickNoise { U = texu, V = texv }, Rhs = new MNConstant { Value = 0.1 } };
            var brickcolor = new MNAdd { Lhs = new MNAdd { Lhs = val05, Rhs = perlinbandscale }, Rhs = bricknoise };
            var mortarcolor = new MNAdd { Lhs = new MNConstant { Value = 0.4 }, Rhs = new MNAdd { Lhs = perlinhighscale, Rhs = perlinlowscale } };
            var colorr = new MNLerp { Value0 = mortarcolor, Value1 = brickcolor, Factor = brickmask };
            var colorg = new MNLerp { Value0 = mortarcolor, Value1 = val0, Factor = brickmask };
            var colorb = new MNLerp { Value0 = mortarcolor, Value1 = val0, Factor = brickmask };
            return new MNVector4D { R = colorr, G = colorg, B = colorb, A = val1 };
        }
        public static IMNNode<Vector4D> MarbleTile()
        {
            return new MNCheckerboard { Color1 = new MNMarbleWhite(), Color2 = new MNMarbleBlack() };
        }
    }
}