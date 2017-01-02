using System;
using System.Drawing;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy
{
    class WireframeGDIPlus : IWireframeRenderer
    {
        public WireframeGDIPlus(DrawingContext drawingContext, int buffer_width, int buffer_height)
        {
            this.drawingContext = drawingContext;
            buffer_bitmap = new Bitmap(buffer_width, buffer_height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
        }
        public void WireframeBegin()
        {
            graphics = Graphics.FromImage(buffer_bitmap);
            graphics.FillRectangle(new System.Drawing.SolidBrush(System.Drawing.Color.Black), new Rectangle(0, 0, buffer_bitmap.Width, buffer_bitmap.Height));
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
            graphics.Dispose();
            graphics = null;
            BitmapSource bitmap_source = Imaging.CreateBitmapSourceFromHBitmap(buffer_bitmap.GetHbitmap(), IntPtr.Zero, System.Windows.Int32Rect.Empty, BitmapSizeOptions.FromWidthAndHeight(buffer_bitmap.Width, buffer_bitmap.Height));
            drawingContext.DrawImage(bitmap_source, new Rect(0, 0, buffer_bitmap.Width, buffer_bitmap.Height));
        }
        private DrawingContext drawingContext = null;
        private System.Drawing.Pen pen = null;
        Bitmap buffer_bitmap = null;
        Graphics graphics = null;
    }
}