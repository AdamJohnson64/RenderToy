using RenderToy.Materials;
using RenderToy.Utility;
using System;

namespace RenderToy.Textures
{
    class Texture24 : IMNNode<Vector4D>, INamed
    {
        public Texture24(string name, int width, int height, byte[] data)
        {
            Name = name;
            Width = width;
            Height = height;
            Data = data;
        }
        public string GetName()
        {
            return Name;
        }
        public bool IsConstant()
        {
            return false;
        }
        public Vector4D Eval(EvalContext context)
        {
            int x = Math.Max(0, Math.Min((int)(context.U * Width), Width - 1));
            int y = Math.Max(0, Math.Min((int)(context.V * Height), Height - 1));
            byte b = Data[0 + 3 * x + 3 * Width * y];
            byte g = Data[1 + 3 * x + 3 * Width * y];
            byte r = Data[2 + 3 * x + 3 * Width * y];
            return new Vector4D(r / 255.0, g / 255.0, b / 255.0, 1);
        }
        public readonly string Name;
        public readonly int Width;
        public readonly int Height;
        public readonly byte[] Data;
    }
}