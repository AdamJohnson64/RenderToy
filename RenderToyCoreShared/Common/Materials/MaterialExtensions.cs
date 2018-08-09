////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Expressions;
using RenderToy.Math;
using RenderToy.PipelineModel;
using RenderToy.Textures;
using System;

namespace RenderToy.Materials
{
    public static class MaterialExtensions
    {
        public const int ThumbnailSize = 32;
        public static void ConvertToBitmap(this IImageBgra32 node, IntPtr bitmapptr, int bitmapWidth, int bitmapHeight, int bitmapstride)
        {
            if (node == null || bitmapptr == IntPtr.Zero) return;
            unsafe
            {
                for (int y = 0; y < bitmapHeight; ++y)
                {
                    void* raster = (byte*)bitmapptr.ToPointer() + bitmapstride * y;
                    for (int x = 0; x < bitmapWidth; ++x)
                    {
                        ((uint*)raster)[x] = node.GetImagePixel(x, y);
                    }
                }
            }
        }
        public static IImageBgra32 GetImageConverter(this IMaterial node, int suggestedWidth, int suggestedHeight)
        {
            if (node == null) return GetImageConverter(StockMaterials.Missing, ThumbnailSize, ThumbnailSize);
            System.Type type = node.GetType();
            if (node is ITexture)
            {
                return GetImageConverter(((ITexture)node).GetSurface(0, 0), suggestedWidth, suggestedHeight);
            }
            else if (node is IImageBgra32)
            {
                return (IImageBgra32)node;
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
                    return Rasterization.ColorToUInt32(new Vector4D(v, v, v, 1));
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
                    return Rasterization.ColorToUInt32(lambda(context));
                });
            }
            else
            {
                return GetImageConverter(StockMaterials.Missing, ThumbnailSize, ThumbnailSize);
            }
        }
        class ImageConverterAdaptor : IImageBgra32
        {
            public ImageConverterAdaptor(int width, int height, Func<int, int, uint> sampler)
            {
                Width = width;
                Height = height;
                Sampler = sampler;
            }
            public bool IsConstant() { return false; }
            public int GetImageWidth() { return Width; }
            public int GetImageHeight() { return Height; }
            public uint GetImagePixel(int x, int y) { return Sampler(x, y); }
            int Width, Height;
            Func<int, int, uint> Sampler;
        }
    }
}