using RenderToyCOM;
using RenderToy.DirectX;
using RenderToy.Math;
using System;

namespace RenderToy.Textures
{
    class SurfaceRegion : ISurface
    {
        public SurfaceRegion(Surface source, int x, int y, int w, int h)
        {
            Source = source;
            X = x;
            Y = y;
            W = w;
            H = h;
        }
        public DXGI_FORMAT GetFormat()
        {
            return Source.GetFormat();
        }
        public int GetImageWidth()
        {
            return W;
        }
        public int GetImageHeight()
        {
            return H;
        }
        public Func<int, int, Vector4D> GetImageReader()
        {
            var reader = Source.GetImageReader();
            return (x, y) => reader(X + x, Y + y);
        }
        public Action<int, int, Vector4D> GetImageWriter()
        {
            var writer = Source.GetImageWriter();
            return (x, y, rgba) => writer(X + x, Y + y, rgba);
        }
        public byte[] Copy()
        {
            int bytesperpixel = Direct3DHelper.GetPixelSize(Source.format);
            byte[] copy = new byte[bytesperpixel * W * H];
            int srcw = Source.GetImageWidth();
            for (int raster = 0; raster < H; ++raster)
            {
                Buffer.BlockCopy(Source.data, bytesperpixel * X + bytesperpixel * srcw * (Y + raster), copy, bytesperpixel * W * raster, bytesperpixel * W);
            }
            return copy;
        }
        public bool IsConstant()
        {
            return false;
        }
        readonly Surface Source;
        int X, Y, W, H;
    }
}