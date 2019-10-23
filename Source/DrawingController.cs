using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace Arcturus.Managed
{
    class DrawingController : FrameworkElement
    {
        protected override void OnRender(System.Windows.Media.DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            base.OnMouseDown(e);
            FakeDocument document = DataContext as FakeDocument;
            if (document == null)
            {
                return;
            }
            var m = e.GetPosition(this);
            int ibest = -1;
            float lbest = 16 * 16;
            for (int i = 0; i < document.points.Length; ++i)
            {
                Vec2 d = new Vec2 { X = (float)(document.points[i].X - m.X), Y = (float)(document.points[i].Y - m.Y) };
                float lthis = d.X * d.X + d.Y * d.Y;
                if (lthis < lbest)
                {
                    ibest = i;
                    lbest = lthis;
                }
            }
            selected = ibest;
        }
        protected override void OnMouseUp(MouseButtonEventArgs e)
        {
            base.OnMouseUp(e);
            selected = -1;
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            FakeDocument document = DataContext as FakeDocument;
            if (document == null)
            {
                return;
            }
            if (selected != -1)
            {
                var m = e.GetPosition(this);
                document.points[selected].X = (float)m.X;
                document.points[selected].Y = (float)m.Y;
                document.Signal();
            }
        }
        int selected = -1;
    }
}