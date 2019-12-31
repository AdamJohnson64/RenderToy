using RenderToyCOM;
using RenderToy.Expressions;
using RenderToy.Math;
using RenderToy.PipelineModel;
using RenderToy.Textures;
using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace RenderToy.Materials
{
    public static class MaterialExtensions
    {
        public const int ThumbnailSize = 32;
        public static void ConvertToBgra32(this ISurface node, IntPtr bitmapptr, int bitmapWidth, int bitmapHeight, int bitmapstride)
        {
            if (node == null || bitmapptr == IntPtr.Zero) return;
            var reader = node.GetImageReader();
            unsafe
            {
                for (int y = 0; y < bitmapHeight; ++y)
                {
                    void* raster = (byte*)bitmapptr.ToPointer() + bitmapstride * y;
                    for (int x = 0; x < bitmapWidth; ++x)
                    {
                        ((uint*)raster)[x] = Rasterization.ColorToUInt32(reader(x, y));
                    }
                }
            }
        }
        public static ConfiguredTaskAwaitable ConvertToBgra32Async(this ISurface node, IntPtr bitmapptr, int bitmapWidth, int bitmapHeight, int bitmapstride)
        {
            return Task.Run(() => ConvertToBgra32(node, bitmapptr, bitmapWidth, bitmapHeight, bitmapstride)).ConfigureAwait(false);
        }
        public static ISurface GetImageConverter(this IMaterial node, int suggestedWidth, int suggestedHeight)
        {
            if (node == null) return GetImageConverter(StockMaterials.Missing, ThumbnailSize, ThumbnailSize);
            System.Type type = node.GetType();
            if (node is ITexture)
            {
                return GetImageConverter(((ITexture)node).GetSurface(0, 0), suggestedWidth, suggestedHeight);
            }
            else if (node is ISurface)
            {
                return (ISurface)node;
            }
            else if (node is IMNNode<double>)
            {
                var lambda = ((IMNNode<double>)node).CompileMSIL();
                var context = new EvalContext();
                return new ImageConverterAdaptor(suggestedWidth, suggestedHeight, (x, y) =>
                {
                    context.U = (x + 0.5) / suggestedWidth;
                    context.V = (y + 0.5) / suggestedHeight;
                    double v = lambda(context);
                    return new Vector4D(v, v, v, 1);
                });
            }
            else if (node is IMNNode<Vector4D>)
            {
                var lambda = ((IMNNode<Vector4D>)node).CompileMSIL();
                var context = new EvalContext();
                return new ImageConverterAdaptor(suggestedWidth, suggestedHeight, (x, y) =>
                {
                    context.U = (x + 0.5) / suggestedWidth;
                    context.V = (y + 0.5) / suggestedHeight;
                    return lambda(context);
                });
            }
            else
            {
                return GetImageConverter(StockMaterials.Missing, ThumbnailSize, ThumbnailSize);
            }
        }
        class ImageConverterAdaptor : ISurface
        {
            public ImageConverterAdaptor(int width, int height, Func<int, int, Vector4D> sampler)
            {
                Width = width;
                Height = height;
                Sampler = sampler;
            }
            public bool IsConstant() { return false; }
            public DXGI_FORMAT GetFormat() { return DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM; }
            public int GetImageWidth() { return Width; }
            public int GetImageHeight() { return Height; }
            public Func<int, int, Vector4D> GetImageReader()
            {
                return (x, y) =>
                {
                    return Sampler(x, y);
                };
            }
            public Action<int, int, Vector4D> GetImageWriter()
            {
                throw new NotSupportedException();
            }
            public byte[] Copy()
            {
                var copy = new byte[4 * Width * Height];
                var reader = GetImageReader();
                for (int y = 0; y < Height; ++y)
                {
                    for (int x = 0; x < Width; ++x)
                    {
                        unsafe
                        {
                            fixed (byte* pixel = &copy[4 * x + 4 * Width * y])
                            {
                                *(uint*)pixel = Rasterization.ColorToUInt32(reader(x,y));
                            }
                        }
                    }
                }
                return copy;
            }
            int Width, Height;
            Func<int, int, Vector4D> Sampler;
        }
    }
}