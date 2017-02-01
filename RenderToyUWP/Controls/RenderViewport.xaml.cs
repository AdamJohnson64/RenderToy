using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Runtime.InteropServices;
using Windows.UI.Input;

// The User Control item template is documented at http://go.microsoft.com/fwlink/?LinkId=234236

namespace RenderToy
{
    public sealed partial class RenderViewport : UserControl
    {
        public RenderViewport()
        {
            this.InitializeComponent();
            Repaint();
        }
        protected override void OnPointerMoved(PointerRoutedEventArgs e)
        {
            base.OnPointerMoved(e);
            if (!isDragging) return;
            PointerPoint dragTo = e.GetCurrentPoint(this);
            double dx = dragTo.Position.X - dragFrom.Position.X;
            double dy = dragTo.Position.Y - dragFrom.Position.Y;
            x += dx * -0.05;
            y += dy * 0.05;
            dragFrom = dragTo;
            Repaint();
        }
        protected override void OnPointerPressed(PointerRoutedEventArgs e)
        {
            base.OnPointerPressed(e);
            CapturePointer(e.Pointer);
            isDragging = true;
            dragFrom = e.GetCurrentPoint(this);
        }
        protected override void OnPointerReleased(PointerRoutedEventArgs e)
        {
            base.OnPointerReleased(e);
            isDragging = false;
            ReleasePointerCapture(e.Pointer);
        }
        void Repaint()
        {
            int RENDER_WIDTH = 512;
            int RENDER_HEIGHT = 512;
            var bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT);
            byte[] image = new byte[4 * RENDER_WIDTH * RENDER_HEIGHT];
            unsafe
            {
                GCHandle handle = GCHandle.Alloc(image, GCHandleType.Pinned);
                try
                {
                    RenderCS.Wireframe(Scene.Default, MathHelp.Invert(MathHelp.CreateMatrixTranslate(x, y, z)) * CameraPerspective.CreateProjection(0.01, 100.0, 45, 45), handle.AddrOfPinnedObject(), RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                }
                finally
                {
                    handle.Free();
                }
            }
            using (var stream = bitmap.PixelBuffer.AsStream())
            {
                stream.Write(image, 0, 4 * RENDER_WIDTH * RENDER_HEIGHT);
            }
            bitmap.Invalidate();
            Component_Image.Source = bitmap;
        }
        bool isDragging = false;
        PointerPoint dragFrom;
        double x = 0;
        double y = 2;
        double z = -5;
    }
}
