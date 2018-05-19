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
                return ConvertToBitmap((IMaterial)value, bitmapWidth, bitmapHeight);
            }
            return null;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
        public static WriteableBitmap ConvertToBitmap(IMaterial node, int bitmapWidth, int bitmapHeight)
        {
            if (node == null) return null;
            System.Type type = node.GetType();
            if (typeof(ITexture).IsAssignableFrom(type))
            {
                return ConvertToBitmap((ITexture)node);
            }
            if (typeof(IImageBgra32).IsAssignableFrom(type))
            {
                return ConvertToBitmap((IImageBgra32)node);
            }
            else if (typeof(IMNNode<double>).IsAssignableFrom(type))
            {
                return ConvertToBitmap((IMNNode<double>)node, bitmapWidth, bitmapHeight);
            }
            else if (typeof(IMNNode<Vector4D>).IsAssignableFrom(type))
            {
                return ConvertToBitmap((IMNNode<Vector4D>)node, bitmapWidth, bitmapHeight);
            }
            return null;
        }
        public static WriteableBitmap ConvertToBitmap(ITexture node)
        {
            return ConvertToBitmap(node.GetTextureLevel(0));
        }
        public static WriteableBitmap ConvertToBitmap(IImageBgra32 node)
        {
            if (node == null) return null;
            int bitmapWidth = node.GetImageWidth();
            int bitmapHeight = node.GetImageHeight();
            var bitmap = new WriteableBitmap(bitmapWidth, bitmapHeight, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            ConvertToBitmap(node, bitmap.BackBuffer, bitmapWidth, bitmapHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmapWidth, bitmapHeight));
            bitmap.Unlock();
            return bitmap;
        }
        public static void ConvertToBitmap(IImageBgra32 node, IntPtr bitmapptr, int bitmapwidth, int bitmapheight, int bitmapstride)
        {
            node.CopyImageTo(bitmapptr, bitmapwidth, bitmapheight, bitmapstride);
        }
        public static WriteableBitmap ConvertToBitmap(IMNNode<double> node, int bitmapWidth, int bitmapHeight)
        {
            if (node == null || bitmapWidth == 0 || bitmapHeight == 0) return null;
            var bitmap = new WriteableBitmap(bitmapWidth, bitmapHeight, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            ConvertToBitmap(node, bitmap.BackBuffer, bitmapWidth, bitmapHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmapWidth, bitmapHeight));
            bitmap.Unlock();
            return bitmap;
        }
        public static void ConvertToBitmap(IMNNode<double> node, IntPtr bitmapptr, int bitmapwidth, int bitmapheight, int bitmapstride)
        {
            var context = new EvalContext();
            unsafe
            {
                for (int y = 0; y < bitmapheight; ++y)
                {
                    void* raster = (byte*)bitmapptr.ToPointer() + bitmapstride * y;
                    for (int x = 0; x < bitmapwidth; ++x)
                    {
                        context.U = (x + 0.5) / bitmapwidth;
                        context.V = (y + 0.5) / bitmapheight;
                        double v = node.Eval(context);
                        ((uint*)raster)[x] = Rasterization.ColorToUInt32(new Vector4D(v, v, v, 1));
                    }
                }
            }
        }
        public static WriteableBitmap ConvertToBitmap(IMNNode<Vector4D> node, int bitmapWidth, int bitmapHeight)
        {
            if (node == null || bitmapWidth == 0 || bitmapHeight == 0) return null;
            var bitmap = new WriteableBitmap(bitmapWidth, bitmapHeight, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            ConvertToBitmap(node, bitmap.BackBuffer, bitmapWidth, bitmapHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmapWidth, bitmapHeight));
            bitmap.Unlock();
            return bitmap;
        }
        public static void ConvertToBitmap(IMNNode<Vector4D> node, IntPtr bitmapptr, int bitmapwidth, int bitmapheight, int bitmapstride)
        {
            var context = new EvalContext();
            unsafe
            {
                for (int y = 0; y < bitmapheight; ++y)
                {
                    void* raster = (byte*)bitmapptr.ToPointer() + bitmapstride * y;
                    for (int x = 0; x < bitmapwidth; ++x)
                    {
                        context.U = (x + 0.5) / bitmapwidth;
                        context.V = (y + 0.5) / bitmapheight;
                        ((uint*)raster)[x] = Rasterization.ColorToUInt32(node.Eval(context));
                    }
                }
            }
        }
        const int bitmapWidth = 256;
        const int bitmapHeight = 256;
        public static int ThumbnailSize = 32;
    }
}