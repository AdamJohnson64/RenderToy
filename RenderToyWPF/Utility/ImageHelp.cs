////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static class ImageHelp
    {
        public delegate ImageSource ImageFunction(Scene scene, Matrix3D mvp, int render_width, int render_height);
        public delegate void FillFunction(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
        public static ImageSource CreateImage(FillFunction fillwith, Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            WriteableBitmap bitmap = new WriteableBitmap(render_width, render_height, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            fillwith(scene, mvp, bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, render_width, render_height));
            bitmap.Unlock();
            return bitmap;
        }
        public static ImageSource CreateImage(ImageFunction fillwith, Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            return fillwith(scene, mvp, render_width, render_height);
        }
        public static void DrawImage(FillFunction fillwith, Scene scene, Matrix3D mvp, int render_width, int render_height, DrawingContext drawingContext, double width, double height)
        {
            var bitmap = CreateImage(fillwith, scene, mvp, render_width, render_height);
            drawingContext.DrawImage(bitmap, new Rect(0, 0, width, height));
        }
        public static void DrawImage(ImageFunction fillwith, Scene scene, Matrix3D mvp, int render_width, int render_height, DrawingContext drawingContext, double width, double height)
        {
            var bitmap = fillwith(scene, mvp, render_width, render_height);
            drawingContext.DrawImage(bitmap, new Rect(0, 0, width, height));
        }
    }
}