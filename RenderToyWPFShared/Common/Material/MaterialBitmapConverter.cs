////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Textures;
using System;
using System.Globalization;
using System.Threading.Tasks;
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
                return ConvertToBitmapAsync(((IMaterial)value).GetImageConverter(256, 256)).Result;
            }
            return null;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
        public static WriteableBitmap ConvertToBitmap(IMaterial node, int suggestedWidth, int suggestedHeight)
        {
            return ConvertToBitmapAsync(node, suggestedWidth, suggestedHeight).Result;
        }
        public static async Task<WriteableBitmap> ConvertToBitmapAsync(IMaterial node, int suggestedWidth, int suggestedHeight)
        {
            return await ConvertToBitmapAsync(node.GetImageConverter(suggestedWidth, suggestedHeight));
        }
        public static WriteableBitmap ConvertToBitmap(IImageBgra32 node)
        {
            return ConvertToBitmapAsync(node).Result;
        }
        public static async Task<WriteableBitmap> ConvertToBitmapAsync(IImageBgra32 node)
        {
            if (node == null) return null;
            var bitmap = new WriteableBitmap(node.GetImageWidth(), node.GetImageHeight(), 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            await node.ConvertToBitmapAsync(bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));
            bitmap.Unlock();
            return bitmap;
        }
    }
}