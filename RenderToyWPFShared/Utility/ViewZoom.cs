using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public class ViewZoom : ContentControl
    {
        public ViewZoom()
        {
            ClipToBounds = true;
        }
        Transform ContentTransform
        {
            get
            {
                var visual = GetVisualChild(0) as UIElement;
                if (visual == null) return Transform.Identity;
                double windowwidth = DesiredSize.Width;
                double windowheight = DesiredSize.Height;
                double contentwidth = visual.DesiredSize.Width;
                double contentheight = visual.DesiredSize.Height;
                double zoomwidth = windowwidth / contentwidth;
                double zoomheight = windowheight / contentheight;
                double zoombest = Math.Min(zoomwidth, zoomheight) * zoom;
                var transform = new TransformGroup();
                transform.Children.Add(new TranslateTransform(-contentwidth / 2, -contentheight / 2));
                transform.Children.Add(new TranslateTransform(-contentwidth * offsetx, -contentheight * offsety));
                transform.Children.Add(new ScaleTransform(zoombest, zoombest));
                transform.Children.Add(new TranslateTransform(windowwidth / 2, windowheight / 2));
                return transform;
            }
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            if (e.Delta < 0) zoom /= 2;
            if (e.Delta > 0) zoom *= 2;
            SetZoom();
            e.Handled = true;
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return;
            var transform = ContentTransform;
            Point tl = transform.Transform(new Point(0, 0));
            Point br = transform.Transform(new Point(visual.DesiredSize.Width, visual.DesiredSize.Height));
            drawingContext.DrawRectangle(null, new Pen(Brushes.Green, 1), new Rect(tl, br));
        }
        void SetZoom()
        {
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return;
            visual.RenderTransform = ContentTransform;
            InvalidateVisual();
        }
        protected override Size ArrangeOverride(Size arrangeSize)
        {
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return new Size(0, 0);
            visual.Arrange(new Rect(0, 0, visual.DesiredSize.Width, visual.DesiredSize.Height));
            SetZoom();
            return DesiredSize;
        }
        protected override Size MeasureOverride(Size constraint)
        {
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return new Size(0, 0);
            visual.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
            return constraint;
        }
        double zoom = 1;
        double offsetx = 0;
        double offsety = 0;
    }
}