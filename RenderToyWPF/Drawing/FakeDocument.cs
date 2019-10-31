using System.ComponentModel;

namespace Arcturus.Managed
{
    public class FakeDocument : INotifyPropertyChanged
    {
        public static FakeDocument Global = new FakeDocument();
        public FakeDocument()
        {
            Signal();
        }
        public Vec2[] points =
        {
            new Vec2 { X = 100.5f, Y = 100.5f },
            new Vec2 { X = 20.5f, Y = 20.5f },
            new Vec2 { X = 180.5f, Y = 20.5f },
            new Vec2 { X = 180.5f, Y = 180.5f },
            new Vec2 { X = 20.5f, Y = 180.5f },
        };
        public void Signal()
        {
            context.reset();
            context.setColor(new Vec4 { X = 0, Y = 0, Z = 0, W = 1 });
            context.setWidth(1);
            // Draw a quad.
            context.moveTo(points[1]);
            for (int i = 2; i < 5; ++i)
            {
                context.lineTo(points[i]);
            }
            context.lineTo(points[1]);
            // Draw a bullseye.
            context.drawCircle(points[0], 40);
            context.fillCircle(points[0], 10);
            // Draw handles.
            context.setWidth(1);
            foreach (var p in points)
            {
                context.setColor(new Vec4 { X = 0.8f, Y = 0.8f, Z = 1, W = 1 });
                context.fillRectangle(new Vec2 { X = p.X - 4, Y = p.Y - 4 }, new Vec2 { X = p.X + 4, Y = p.Y + 4 });
                context.setColor(new Vec4 { X = 0, Y = 0, Z = 0, W = 1 });
                context.drawRectangle(new Vec2 { X = p.X - 4, Y = p.Y - 4 }, new Vec2 { X = p.X + 4, Y = p.Y + 4 });
            }
            // Notify all listeners of update.
            if (PropertyChanged != null)
            {
                PropertyChanged(this, null);
            }
        }
        public DrawingContext context = new DrawingContext();
        public event PropertyChangedEventHandler PropertyChanged;
    };
}