////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.PipelineModel;
using RenderToy.Textures;
using RenderToy.Utility;
using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    public class MaterialBitmapConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is IMaterial)
            {
                return ConvertToBitmap(GetImageConverter((IMaterial)value, 256, 256));
            }
            return null;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
        public static WriteableBitmap ConvertToBitmap(IMaterial node, int suggestedWidth, int suggestedHeight)
        {
            return ConvertToBitmap(GetImageConverter(node, suggestedWidth, suggestedHeight));
        }
        public static WriteableBitmap ConvertToBitmap(IImageBgra32 node)
        {
            if (node == null) return null;
            var bitmap = new WriteableBitmap(node.GetImageWidth(), node.GetImageHeight(), 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            ConvertToBitmap(node, bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));
            bitmap.Unlock();
            return bitmap;
        }
        public static void ConvertToBitmap(IImageBgra32 node, IntPtr bitmapptr, int bitmapWidth, int bitmapHeight, int bitmapstride)
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
        public static IImageBgra32 GetImageConverter(IMaterial node, int suggestedWidth, int suggestedHeight)
        {
            if (node == null) return GetImageConverter(StockMaterials.Missing, ThumbnailSize, ThumbnailSize);
            System.Type type = node.GetType();
            if (typeof(ITexture).IsAssignableFrom(type))
            {
                return GetImageConverter(((ITexture)node).GetTextureLevel(0), suggestedWidth, suggestedHeight);
            }
            else if (typeof(IImageBgra32).IsAssignableFrom(type))
            {
                return (IImageBgra32)node;
            }
            else if (typeof(IMNNode<double>).IsAssignableFrom(type))
            {
                var convert = (IMNNode<double>)node;
                var context = new EvalContext();
                var param = System.Linq.Expressions.Expression.Parameter(typeof(EvalContext));
                var body = convert.CreateExpression(param);
                var lambda = System.Linq.Expressions.Expression.Lambda<Func<EvalContext, double>>(body, param).Compile();
                return new ImageConverterAdaptor(suggestedWidth, suggestedHeight, (x, y) =>
                {
                    context.U = (x + 0.5) / suggestedWidth;
                    context.V = (y + 0.5) / suggestedHeight;
                    double v = lambda(context);
                    return Rasterization.ColorToUInt32(new Vector4D(v, v, v, 1));
                });
            }
            else if (typeof(IMNNode<Vector4D>).IsAssignableFrom(type))
            {
                var convert = (IMNNode<Vector4D>)node;
                var context = new EvalContext();
                var param = System.Linq.Expressions.Expression.Parameter(typeof(EvalContext));
                var body = VisitorTest.Reduce(convert.CreateExpression(param));
                var lambda = System.Linq.Expressions.Expression.Lambda<Func<EvalContext, Vector4D>>(body, param).Compile();
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
        public static int ThumbnailSize = 32;
    }
}