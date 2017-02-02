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
using Windows.UI.Core;
using Windows.System;

// The User Control item template is documented at http://go.microsoft.com/fwlink/?LinkId=234236

namespace RenderToy
{
    public sealed class RenderViewport : UserControl
    {
        Image control_image = new Image();
        TransformPosQuat Camera = new TransformPosQuat { Position = new Point3D(0, 2, -5) };
        public RenderViewport()
        {
            SizeChanged += (s, e) => { Repaint(); };
            Content = control_image;
        }
        protected override void OnPointerMoved(PointerRoutedEventArgs e)
        {
            base.OnPointerMoved(e);
            if (!isDragging) return;
            PointerPoint dragTo = e.GetCurrentPoint(this);
            double dx = dragTo.Position.X - dragFrom.Position.X;
            double dy = dragTo.Position.Y - dragFrom.Position.Y;
            dragFrom = dragTo;
            // Detect modifier keys.
            var stateLeftControl = CoreWindow.GetForCurrentThread().GetKeyState(VirtualKey.LeftControl);
            var stateLeftShift = CoreWindow.GetForCurrentThread().GetKeyState(VirtualKey.LeftShift);
            bool isPressedLeftControl = (stateLeftControl & CoreVirtualKeyStates.Down) == CoreVirtualKeyStates.Down;
            bool isPressedLeftShift = (stateLeftShift & CoreVirtualKeyStates.Down) == CoreVirtualKeyStates.Down;
            // Process camera motion with modifier keys.
            if (isPressedLeftShift && isPressedLeftControl)
            {
                // Truck Mode (CTRL + SHIFT).
                Camera.TranslatePost(new Vector3D(0, 0, dy * -0.05));
            }
            else if (!isPressedLeftShift && isPressedLeftControl)
            {
                // Rotate Mode (CTRL Only)
                Camera.RotatePre(new Quaternion(new Vector3D(0, 1, 0), dx * 0.05));
                Camera.RotatePost(new Quaternion(new Vector3D(1, 0, 0), dy * 0.05));
            }
            else if (!isPressedLeftShift && !isPressedLeftControl)
            {
                // Translation Mode (no modifier keys).
                Camera.TranslatePost(new Vector3D(dx * -0.05, dy * 0.05, 0));
            }
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
            int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth) / 4;
            int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight) / 4;
            if (RENDER_WIDTH < 8 || RENDER_HEIGHT < 8) return;
            var mvp = MathHelp.Invert(Camera.Transform) * CameraPerspective.CreateProjection(0.01, 100.0, 45, 45) * CameraPerspective.AspectCorrectFit(ActualWidth, ActualHeight);
            var inverse_mvp = MathHelp.Invert(mvp);
            var bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT);
            byte[] buffer_image = new byte[4 * RENDER_WIDTH * RENDER_HEIGHT];
            byte[] buffer_inverse_mvp = SceneFormatter.CreateFlatMemoryF64(inverse_mvp);
            byte[] buffer_scene = SceneFormatter.CreateFlatMemoryF64(Scene.Default);
            //GCHandle handle = GCHandle.Alloc(buffer_image, GCHandleType.Pinned);
            //RenderCS.Wireframe(Scene.Default, mvp, handle.AddrOfPinnedObject(), RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
            //handle.Free();
            RenderToyCX.RaytraceCPUF64(buffer_scene, buffer_inverse_mvp, buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
            using (var stream = bitmap.PixelBuffer.AsStream())
            {
                stream.Write(buffer_image, 0, 4 * RENDER_WIDTH * RENDER_HEIGHT);
            }
            bitmap.Invalidate();
            control_image.Source = bitmap;
        }
        bool isDragging = false;
        PointerPoint dragFrom;
        double x = 0;
        double y = 2;
        double z = -5;
    }
}
