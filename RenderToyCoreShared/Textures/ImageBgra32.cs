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
    public interface IImageBgra32
    {
        int GetImageWidth();
        int GetImageHeight();
        void CopyImageTo(IntPtr buffer, int width, int height, int stride);
    }
    class ImageBgra32 : IMNNode<Vector4D>, IImageBgra32, INamed
    {
        public ImageBgra32(string name, int width, int height, byte[] data)
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
        public int GetImageWidth()
        {
            return Width;
        }
        public int GetImageHeight()
        {
            return Height;
        }
        public void CopyImageTo(IntPtr buffer, int width, int height, int stride)
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
        public ImageBgra32 BoxFilter()
        {
            // If we have an odd size texture then we can't box filter; skip with null.
            if ((Width & 1) == 1 || (Height & 1) == 1) return null;
            int newwidth = Width >> 1;
            int newheight = Height >> 1;
            byte[] newdata = new byte[4 * newwidth * newheight];
            for (int y = 0; y < newheight; ++y)
            {
                for (int x = 0; x < newwidth; ++x)
                {
                    // Sum the 2x2 block of pixels then average it.
                    var sum = new Vector4D();
                    for (int yb = 0; yb < 2; ++yb)
                    {
                        for (int xb = 0; xb < 2; ++xb)
                        {
                            int xs = x * 2 + xb;
                            int ys = y * 2 + yb;
                            byte r = Data[0 + 4 * xs + 4 * Width * ys];
                            byte g = Data[1 + 4 * xs + 4 * Width * ys];
                            byte b = Data[2 + 4 * xs + 4 * Width * ys];
                            byte a = Data[3 + 4 * xs + 4 * Width * ys];
                            sum = sum + new Vector4D(r / 255.0, g / 255.0, b / 255.0, a / 255.0);
                        }
                    }
                    sum = sum * 0.25;
                    // Regenerate the Rgba32 value and set it into the filtered image.
                    byte rn = (byte)(sum.X * 255.0);
                    byte gn = (byte)(sum.Y * 255.0);
                    byte bn = (byte)(sum.Z * 255.0);
                    byte an = (byte)(sum.W * 255.0);
                    newdata[0 + 4 * x + 4 * newwidth * y] = rn;
                    newdata[1 + 4 * x + 4 * newwidth * y] = gn;
                    newdata[2 + 4 * x + 4 * newwidth * y] = bn;
                    newdata[3 + 4 * x + 4 * newwidth * y] = an;
                }
            }
            return new ImageBgra32(null, newwidth, newheight, newdata);
        }
        public readonly string Name;
        public readonly int Width;
        public readonly int Height;
        public readonly byte[] Data;
    }
}