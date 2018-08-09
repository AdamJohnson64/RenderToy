////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;

namespace RenderToy.WPF
{
    /// <summary>
    /// Render the contents of a D3DImage into a control.
    /// This class should be used if you are rendering to a D3D9 surface.
    /// </summary>
    public class ViewD3DImage : FrameworkElement
    {
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(Target, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            return base.MeasureOverride(availableSize);
        }
        protected D3DImage Target = new D3DImage();
    }
}