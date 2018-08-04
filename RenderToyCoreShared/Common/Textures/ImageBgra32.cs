////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Utility;

namespace RenderToy.Textures
{
    public interface IImageBgra32 : IMaterial
    {
        int GetImageWidth();
        int GetImageHeight();
        uint GetImagePixel(int x, int y);
    }
    public class ImageBgra32 : IImageBgra32, INamed
    {
        public ImageBgra32(string name, int width, int height, byte[] data)
        {
            this.name = name;
            this.width = width;
            this.height = height;
            this.data = data;
        }
        public string Name { get { return name; } }
        public bool IsConstant() { return false; }
        public int GetImageWidth() { return width; }
        public int GetImageHeight() { return height; }
        public uint GetImagePixel(int x, int y)
        {
            int off = 4 * x + 4 * width * y;
            return ((uint)data[off + 0] << 0) | ((uint)data[off + 1] << 8) | ((uint)data[off + 2] << 16) | ((uint)data[off + 3] << 24);
        }
        public ImageBgra32 BoxFilter()
        {
            // If we have an odd size texture then we can't box filter; skip with null.
            if ((width & 1) == 1 || (height & 1) == 1) return null;
            int newwidth = width >> 1;
            int newheight = height >> 1;
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
                            byte r = data[0 + 4 * xs + 4 * width * ys];
                            byte g = data[1 + 4 * xs + 4 * width * ys];
                            byte b = data[2 + 4 * xs + 4 * width * ys];
                            byte a = data[3 + 4 * xs + 4 * width * ys];
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
        public readonly string name;
        public readonly int width;
        public readonly int height;
        public readonly byte[] data;
    }
}