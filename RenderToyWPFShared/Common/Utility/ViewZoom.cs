////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public class ViewZoom : ContentControl
    {
        #region - Section : Construction -
        public ViewZoom()
        {
            ClipToBounds = true;
            CommandBindings.Add(new CommandBinding(RecenterCommand, Recenter));
        }
        #endregion
        #region - Section : Transform Management -
        Transform ContentTransform
        {
            get
            {
                return GenerateTransform(offsetx, offsety, zoom);
            }
        }
        Transform GenerateTransform(double offsetx, double offsety, double zoom)
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
        void UpdateTransform()
        {
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return;
            visual.RenderTransform = ContentTransform;
            InvalidateVisual();
        }
        double zoom = 1;
        double offsetx = 0;
        double offsety = 0;
        bool dragActive = false;
        Point dragPoint;
        #endregion
        #region - Section : Commands -
        public static readonly RoutedUICommand RecenterCommand = new RoutedUICommand { Text = "Recenter" };
        static void Recenter(object sender, System.Windows.Input.ExecutedRoutedEventArgs e)
        {
            var host = (ViewZoom)sender;
            host.zoom = 1;
            host.offsetx = 0;
            host.offsety = 0;
            host.UpdateTransform();
        }
        #endregion
        #region - Section : Overrides -
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!dragActive) return;
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return;
            var pointmouse = e.GetPosition(this);
            pointmouse = GenerateTransform(0, 0, zoom).Inverse.Transform(pointmouse);
            // Recalculate the offsets to move dragPoint to the mouse location.
            offsetx = dragPoint.X - (pointmouse.X / visual.DesiredSize.Width - 0.5);
            offsety = dragPoint.Y - (pointmouse.Y / visual.DesiredSize.Height - 0.5);
            UpdateTransform();
            e.Handled = true;
        }
        protected override void OnMouseRightButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseRightButtonDown(e);
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return;
            var pointmouse = e.GetPosition(this);
            dragPoint = ContentTransform.Inverse.Transform(pointmouse);
            dragPoint.X = dragPoint.X / visual.DesiredSize.Width - 0.5;
            dragPoint.Y = dragPoint.Y / visual.DesiredSize.Height - 0.5;
            dragActive = true;
            CaptureMouse();
            e.Handled = true;
        }
        protected override void OnMouseRightButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseRightButtonUp(e);
            if (!dragActive) return;
            dragActive = false;
            dragPoint = default(Point);
            ReleaseMouseCapture();
            e.Handled = true;
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            if (e.Delta < 0) zoom /= 2;
            if (e.Delta > 0) zoom *= 2;
            UpdateTransform();
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
        protected override Size ArrangeOverride(Size arrangeSize)
        {
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return new Size(0, 0);
            visual.Arrange(new Rect(0, 0, visual.DesiredSize.Width, visual.DesiredSize.Height));
            UpdateTransform();
            return DesiredSize;
        }
        protected override Size MeasureOverride(Size constraint)
        {
            var visual = GetVisualChild(0) as UIElement;
            if (visual == null) return new Size(0, 0);
            visual.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
            return constraint;
        }
        #endregion
    }
}