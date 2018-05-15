////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System.Windows;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public abstract class View3DBase : FrameworkElement
    {
        #region - Section : Construction -
        public View3DBase()
        {
            SetBinding(CameraTransformProperty, new Binding { RelativeSource = new RelativeSource(RelativeSourceMode.Self), Path = new PropertyPath("Camera.Transform") });
            SetBinding(SceneProperty, new Binding { RelativeSource = new RelativeSource(RelativeSourceMode.Self), Path = new PropertyPath(DataContextProperty) });
        }
        #endregion
        #region - Section : DependencyProperties -
        public static DependencyProperty SceneProperty = DependencyProperty.Register("Scene", typeof(IScene), typeof(View3DBase), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender, (d, e) => { ((View3DBase)d).OnSceneChanged((IScene)e.NewValue); }));
        public IScene Scene
        {
            get { return (IScene)GetValue(SceneProperty); }
            set { SetValue(SceneProperty, value); }
        }
        protected virtual void OnSceneChanged(IScene scene)
        {
        }
        public static DependencyProperty CameraProperty = DependencyProperty.Register("Camera", typeof(Camera), typeof(View3DBase));
        public Camera Camera
        {
            get { return (Camera)GetValue(CameraProperty); }
            set { SetValue(CameraProperty, value); }
        }
        public static DependencyProperty CameraTransformProperty = DependencyProperty.Register("CameraTransform", typeof(Matrix3D), typeof(View3DBase), new FrameworkPropertyMetadata(Matrix3D.Identity, FrameworkPropertyMetadataOptions.AffectsRender));
        public Matrix3D CameraTransform
        {
            get { return (Matrix3D)GetValue(CameraTransformProperty); }
        }
        #endregion
        #region - Section : Camera -
        protected Matrix3D View
        {
            get
            {
                return MathHelp.Invert(CameraTransform);
            }
        }
        protected Matrix3D Projection
        {
            get
            {
                return CameraMat.Projection;
            }
        }
        protected Matrix3D MVP
        {
            get
            {
                return View * Projection * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            }
        }
        Perspective CameraMat = new Perspective();
        #endregion
        #region - Section : Rendering -
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            OnRenderToy(drawingContext);
        }
        protected abstract void OnRenderToy(DrawingContext drawingContext);
        #endregion
        #region - Overrides : UIElement -
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            CaptureMouse();
            Mouse.OverrideCursor = Cursors.None;
            isDragging = true;
            dragFrom = System.Windows.Forms.Cursor.Position;
        }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            Mouse.OverrideCursor = null;
            ReleaseMouseCapture();
            isDragging = false;
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!isDragging) return;
            System.Drawing.Point dragTo = System.Windows.Forms.Cursor.Position;
            double dx = dragTo.X - dragFrom.X;
            double dy = dragTo.Y - dragFrom.Y;
            System.Windows.Forms.Cursor.Position = dragFrom;
            // If there's no camera then there's nothing to update from here.
            if (Camera == null) return;
            // Detect modifier keys.
            bool isPressedLeftControl = Keyboard.IsKeyDown(Key.LeftCtrl);
            bool isPressedLeftShift = Keyboard.IsKeyDown(Key.LeftShift);
            // Process camera motion with modifier keys.
            if (isPressedLeftShift && isPressedLeftControl)
            {
                // Truck Mode (CTRL + SHIFT).
                Camera.Object.TranslatePost(new Vector3D(0, 0, dy * -0.05));
            }
            else if (!isPressedLeftShift && isPressedLeftControl)
            {
                // Rotate Mode (CTRL Only)
                Camera.Object.RotatePre(new Quaternion(new Vector3D(0, 1, 0), dx * 0.05));
                Camera.Object.RotatePost(new Quaternion(new Vector3D(1, 0, 0), dy * 0.05));
            }
            else if (!isPressedLeftShift && !isPressedLeftControl)
            {
                // Translation Mode (no modifier keys).
                Camera.Object.TranslatePost(new Vector3D(dx * -0.05, dy * 0.05, 0));
            }
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            Camera.Object.TranslatePost(new Vector3D(0, 0, e.Delta * 0.01));
        }
        bool isDragging = false;
        System.Drawing.Point dragFrom;
        #endregion
    }
}