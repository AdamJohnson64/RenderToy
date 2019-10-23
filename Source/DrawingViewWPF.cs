using System.ComponentModel;
using System.Windows;
using System.Windows.Media;

namespace Arcturus.Managed
{
    class DrawingViewWPF : FrameworkElement
    {
        public DrawingViewWPF()
        {
            ClipToBounds = true;
            DataContextChanged += (s, e) =>
            {
                if (e.OldValue is FakeDocument doc1)
                {
                    doc1.PropertyChanged -= DocumentChanged;
                }
                if (e.NewValue is FakeDocument doc2)
                {
                    doc2.PropertyChanged += DocumentChanged;
                }
            };
        }
        void DocumentChanged(object sender, PropertyChangedEventArgs args)
        {
            InvalidateVisual();
        }
        protected override void OnRender(System.Windows.Media.DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            FakeDocument document = DataContext as FakeDocument;
            if (document == null)
            {
                return;
            }
            unsafe
            {
                var vertices = (Vertex*)document.context.vertexPointer().ToPointer();
                var indexCount = document.context.indexCount();
                var indices = (uint*)document.context.indexPointer().ToPointer();
                uint tricount = indexCount / 3;
                for (uint t = 0; t < tricount; ++t)
                {
                    for (uint i = 0; i < 3; ++i)
                    {
                        var i0 = t * 3 + i;
                        var i1 = t * 3 + ((i + 1) % 3);
                        var v0 = vertices[(int)indices[i0]];
                        var v1 = vertices[(int)indices[i1]];
                        drawingContext.DrawLine(pen, new Point { X = v0.Position.X, Y = v0.Position.Y }, new Point { X = v1.Position.X, Y = v1.Position.Y });
                    }
                }
            }
        }
        static Pen pen = new Pen(Brushes.Black, 1);
    }

}