////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Utility;
using System;
using System.Runtime.InteropServices;

namespace RenderToy.Textures
{
    public interface ITexture
    {
        int GetTextureWidth();
        int GetTextureHeight();
        void CopyTextureTo(IntPtr buffer, int width, int height, int stride);
    }
    class Texture24 : IMNNode<Vector4D>, ITexture, INamed
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
            byte r = Data[2 + 4 * x + 4 * Width * y];
            byte g = Data[1 + 4 * x + 4 * Width * y];
            byte b = Data[0 + 4 * x + 4 * Width * y];
            byte a = Data[3 + 4 * x + 4 * Width * y];
            return new Vector4D(r / 255.0, g / 255.0, b / 255.0, a / 255.0);
        }
        public int GetTextureWidth()
        {
            return Width;
        }
        public int GetTextureHeight()
        {
            return Height;
        }
        public void CopyTextureTo(IntPtr buffer, int width, int height, int stride)
        {
            unsafe
            {
                void* bufferin = Marshal.UnsafeAddrOfPinnedArrayElement(Data, 0).ToPointer();
                void* bufferout = buffer.ToPointer();
                for (int y = 0; y < height; ++y)
                {
                    void* rasterin = (byte*)bufferin + 4 * Width * y;
                    void* rasterout = (byte*)bufferout + stride * y;
                    Buffer.MemoryCopy(rasterin, rasterout, 4 * Width, 4 * Width);
                }
            }
        }
        public readonly string Name;
        public readonly int Width;
        public readonly int Height;
        public readonly byte[] Data;
    }
}