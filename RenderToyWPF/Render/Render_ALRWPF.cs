////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static partial class Render
    {
        #region - Section : Phase 2 - Wireframe Rendering (WPF) -
        public static void WireframeWPF(Scene scene, Matrix3D mvp, DrawingContext drawingContext, double width, double height)
        {
            DrawWireframeCommon(scene, mvp, new WireframeWPF(drawingContext), width, height);
        }
        #endregion
    }
    class WireframeWPF : IWireframeRenderer
    {
        public WireframeWPF(DrawingContext drawingContext)
        {
            this.drawingContext = drawingContext;
            WireframeColor(1, 1, 1);
        }
        public void WireframeBegin()
        {
        }
        public void WireframeColor(double r, double g, double b)
        {
            pen = new Pen(new SolidColorBrush(Color.FromRgb((byte)(r * 255.0), (byte)(g * 255.0), (byte)(b * 255.0))), -1);
        }
        public void WireframeLine(double x1, double y1, double x2, double y2)
        {
            drawingContext.DrawLine(pen, new Point(x1, y1), new Point(x2, y2));
        }
        public void WireframeEnd()
        {
        }
        private DrawingContext drawingContext;
        private Pen pen;
    }
}