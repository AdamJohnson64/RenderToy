////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.Textures;
using RenderToyCOM;
using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.Materials
{
    class RenderTargetBitmapMaterial : ISurface
    {
        public RenderTargetBitmapMaterial(FrameworkElement visual)
        {
            Visual = visual;
        }
        public bool IsConstant()
        {
            return false;
        }
        public DXGI_FORMAT GetFormat()
        {
            return DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM;
        }
        public int GetImageWidth()
        {
            int width = 0;
            Application.Current.Dispatcher.Invoke(() =>
            {
                width = (int)Visual.ActualWidth;
            });
            return width;
        }
        public int GetImageHeight()
        {
            int height = 0;
            Application.Current.Dispatcher.Invoke(() =>
            {
                height = (int)Visual.ActualHeight;
            });
            return height;
        }
        public Func<int, int, Vector4D> GetImageReader()
        {
            return (x, y) =>
            {
                return ((x + y) % 2 == 0) ? StockMaterials.Black : StockMaterials.White;
            };
        }
        public Action<int, int, Vector4D> GetImageWriter()
        {
            throw new NotImplementedException();
        }
        public byte[] Copy()
        {
            byte[] copy = null;
            Application.Current.Dispatcher.Invoke(() =>
            {
                int width = GetImageWidth();
                int height = GetImageHeight();
                var rtb = new RenderTargetBitmap(width, height, 0, 0, PixelFormats.Pbgra32);
                rtb.Render(Visual);
                copy = new byte[4 * width * height];
                rtb.CopyPixels(copy, 4 * width, 0);
            });
            return copy;
        }
        readonly FrameworkElement Visual;
    }
}