////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToyCOM;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Utility;
using System;

namespace RenderToy.Textures
{
    public interface ISurface : IMaterial
    {
        DXGI_FORMAT GetFormat();
        int GetImageWidth();
        int GetImageHeight();
        Func<int, int, Vector4D> GetImageReader();
        Action<int, int, Vector4D> GetImageWriter();
        byte[] Copy();
    }
    public class Surface : ISurface, INamed
    {
        public Surface(string name, DXGI_FORMAT format, int width, int height, byte[] data)
        {
            this.name = name;
            this.format = format;
            this.width = width;
            this.height = height;
            this.data = data;
        }
        public string Name { get { return name; } }
        public bool IsConstant() { return false; }
        public DXGI_FORMAT GetFormat() { return format; }
        public int GetImageWidth() { return width; }
        public int GetImageHeight() { return height; }
        public Func<int, int, Vector4D> GetImageReader()
        {
            switch (format)
            {
                case DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT:
                    return (x, y) =>
                    {
                        unsafe
                        {
                            fixed (byte* ptrByte = &data[4 * 3 * x + 4 * 3 * width * y])
                            {
                                float* ptrFloat = (float*)ptrByte;
                                float r = ptrFloat[0];
                                float g = ptrFloat[1];
                                float b = ptrFloat[2];
                                return new Vector4D(r, g, b, 1);
                            }
                        }
                    };
                case DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM:
                    return (x, y) =>
                    {
                        int off = 4 * x + 4 * width * y;
                        return new Vector4D(data[off + 2] / 255.0, data[off + 1] / 255.0, data[off + 0] / 255.0, data[off + 3] / 255.0);
                    };
                case DXGI_FORMAT.DXGI_FORMAT_B8G8R8X8_UNORM:
                    return (x, y) =>
                    {
                        int off = 4 * x + 4 * width * y;
                        return new Vector4D(data[off + 2] / 255.0, data[off + 1] / 255.0, data[off + 0] / 255.0, 1);
                    };
                default:
                    throw new NotSupportedException("Cannot read " + format + " format textures.");
        }
    }
        public Action<int, int, Vector4D> GetImageWriter()
        {
            switch (format)
            {
                case DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM:
                    return (x, y, rgba) =>
                    {
                        int off = 4 * x + 4 * width * y;
                        data[off + 0] = (byte)(rgba.Z * 255.0);
                        data[off + 1] = (byte)(rgba.Y * 255.0);
                        data[off + 2] = (byte)(rgba.X * 255.0);
                        data[off + 3] = (byte)(rgba.W * 255.0);
                    };
                default:
                    throw new NotSupportedException("Cannot write " + format + " format textures.");
            }
        }
        public Surface BoxFilter()
        {
            if (!(format == DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT.DXGI_FORMAT_B8G8R8X8_UNORM))
            {
                throw new NotSupportedException("Cannot box filter a " + format + " format image.");
            }
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
            return new Surface(null, DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, newwidth, newheight, newdata);
        }
        public byte[] Copy()
        {
            byte[] copy = new byte[data.Length];
            Array.Copy(data, copy, data.Length);
            return copy;
        }
        public readonly string name;
        public readonly DXGI_FORMAT format;
        public readonly int width;
        public readonly int height;
        public readonly byte[] data;
    }
}