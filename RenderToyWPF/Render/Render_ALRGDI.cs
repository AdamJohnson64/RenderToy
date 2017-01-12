////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Drawing;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy
{
    class WireframeGDI : IWireframeRenderer
    {
        public WireframeGDI(DrawingContext drawingContext, int buffer_width, int buffer_height)
        {
            this.drawingContext = drawingContext;
            bitmap = new Bitmap(buffer_width, buffer_height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        }
        public void WireframeBegin()
        {
            if (graphics != null) throw new Exception("Wireframe rendering already entered.");
            graphics = Graphics.FromImage(bitmap);
        }
        public void WireframeColor(double r, double g, double b)
        {
            pen = new System.Drawing.Pen(new SolidBrush(System.Drawing.Color.FromArgb((int)(r * 255.0), (int)(g * 255.0), (int)(b * 255.0))));
        }
        public void WireframeLine(double x1, double y1, double x2, double y2)
        {
            graphics.DrawLine(pen, (float)x1, (float)y1, (float)x2, (float)y2);
        }
        public void WireframeEnd()
        {
            if (graphics == null) throw new Exception("Wireframe rendering not entered.");
            graphics.Dispose();
            graphics = null;
            IntPtr handle = bitmap.GetHbitmap();
            BitmapSource bitmapsource = Imaging.CreateBitmapSourceFromHBitmap(handle, IntPtr.Zero, System.Windows.Int32Rect.Empty, BitmapSizeOptions.FromWidthAndHeight(bitmap.Width, bitmap.Height));
            drawingContext.DrawImage(bitmapsource, new Rect(0, 0, bitmap.Width, bitmap.Height));
            DeleteObject(handle);
        }
        private DrawingContext drawingContext = null;
        private Bitmap bitmap = null;
        private Graphics graphics = null;
        private System.Drawing.Pen pen = null;

        [System.Runtime.InteropServices.DllImport("gdi32.dll")]
        public static extern bool DeleteObject(IntPtr hObject);
    }
}