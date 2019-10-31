////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using System.Windows;
using System.Windows.Input;

namespace RenderToy.WPF
{
    public static class AttachedCamera
    {
        public static DependencyProperty CameraProperty = DependencyProperty.RegisterAttached("Camera", typeof(Camera), typeof(AttachedCamera), new FrameworkPropertyMetadata(null, OnCameraChange));
        public static Camera GetCamera(DependencyObject on)
        {
            return (Camera)on.GetValue(CameraProperty);
        }
        public static void SetCamera(DependencyObject on, Camera camera)
        {
            on.SetValue(CameraProperty, camera);
        }
        public static void OnCameraChange(DependencyObject sender, DependencyPropertyChangedEventArgs propertyevent)
        {
            bool isDragging = false;
            System.Drawing.Point dragFrom = default(System.Drawing.Point);
            var attachingto = sender as FrameworkElement;
            if (attachingto == null) return;
            attachingto.MouseLeftButtonDown += (mousesender, mouseevent) =>
            {
                var frameworkelement = mousesender as FrameworkElement;
                if (frameworkelement == null) return;
                frameworkelement.CaptureMouse();
                Mouse.OverrideCursor = Cursors.None;
                isDragging = true;
                dragFrom = System.Windows.Forms.Cursor.Position;
                mouseevent.Handled = true;
            };
            attachingto.MouseLeftButtonUp += (mousesender, mouseevent) =>
            {
                var frameworkelement = mousesender as FrameworkElement;
                if (frameworkelement == null) return;
                Mouse.OverrideCursor = null;
                frameworkelement.ReleaseMouseCapture();
                isDragging = false;
                mouseevent.Handled = true;
            };
            attachingto.MouseMove += (mousesender, mouseevent) =>
            {
                if (!isDragging) return;
                var senderdependencyobject = mousesender as DependencyObject;
                if (senderdependencyobject == null) return;
                System.Drawing.Point dragTo = System.Windows.Forms.Cursor.Position;
                double dx = dragTo.X - dragFrom.X;
                double dy = dragTo.Y - dragFrom.Y;
                System.Windows.Forms.Cursor.Position = dragFrom;
                // If there's no camera then there's nothing to update from here.
                var camera = GetCamera(senderdependencyobject);
                if (camera == null) return;
                // Detect modifier keys.
                bool isPressedLeftControl = Keyboard.IsKeyDown(Key.LeftCtrl);
                bool isPressedLeftShift = Keyboard.IsKeyDown(Key.LeftShift);
                // Process camera motion with modifier keys.
                if (isPressedLeftShift && isPressedLeftControl)
                {
                    // Truck Mode (CTRL + SHIFT).
                    camera.TranslatePost(new Vector3D(0, 0, dy * -0.05));
                }
                else if (!isPressedLeftShift && isPressedLeftControl)
                {
                    // Rotate Mode (CTRL Only)
                    camera.RotatePre(new Quaternion(new Vector3D(0, 1, 0), dx * 0.05));
                    camera.RotatePost(new Quaternion(new Vector3D(1, 0, 0), dy * 0.05));
                }
                else if (!isPressedLeftShift && !isPressedLeftControl)
                {
                    // Translation Mode (no modifier keys).
                    camera.TranslatePost(new Vector3D(dx * -0.05, dy * 0.05, 0));
                }
                mouseevent.Handled = true;
            };
            attachingto.MouseWheel += (mousesender, mouseevent) =>
            {
                var senderdependencyobject = mousesender as DependencyObject;
                if (senderdependencyobject == null) return;
                var Camera = GetCamera(senderdependencyobject);
                if (Camera == null) return;
                Camera.TranslatePost(new Vector3D(0, 0, mouseevent.Delta * 0.01));
                mouseevent.Handled = true;
            };
        }
    }
}