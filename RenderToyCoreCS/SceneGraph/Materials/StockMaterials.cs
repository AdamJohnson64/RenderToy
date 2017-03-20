////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy.SceneGraph.Materials
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
    }
}