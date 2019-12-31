using RenderToy.Materials;
using RenderToy.Textures;
using System;
using System.Globalization;
using System.Runtime.CompilerServices;
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
                return ConvertToBitmap(((IMaterial)value).GetImageConverter(256, 256));
            }
            return null;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
        public static WriteableBitmap ConvertToBitmap(IMaterial material, int suggestedWidth, int suggestedHeight)
        {
            return ConvertToBitmap(material.GetImageConverter(suggestedWidth, suggestedHeight));
        }
        public static ConfiguredTaskAwaitable<WriteableBitmap> ConvertToBitmapAsync(IMaterial material, int suggestedWidth, int suggestedHeight)
        {
            return Task.Run(() => ConvertToBitmap(material, suggestedWidth, suggestedHeight)).ConfigureAwait(false);
        }
        public static WriteableBitmap ConvertToBitmap(ISurface surface)
        {
            if (surface == null) return null;
            WriteableBitmap bitmap = null;
            Application.Current.Dispatcher.Invoke(() =>
            {
                bitmap = new WriteableBitmap(surface.GetImageWidth(), surface.GetImageHeight(), 0, 0, PixelFormats.Bgra32, null);
                bitmap.Lock();
            });
            surface.ConvertToBgra32(bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
            Application.Current.Dispatcher.Invoke(() =>
            {
                bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));
                bitmap.Unlock();
            });
            return bitmap;
        }
        public static ConfiguredTaskAwaitable<WriteableBitmap> ConvertToBitmapAsync(ISurface surface)
        {
            return Task.Run(() => ConvertToBitmap(surface)).ConfigureAwait(false);
        }
    }
}