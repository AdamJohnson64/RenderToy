////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.Math;
using RenderToy.Transforms;
using System.Windows;

namespace RenderToy.WPF
{
    public class Camera : DependencyObject
    {
        public static DependencyProperty TransformCameraProperty = DependencyProperty.Register("TransformCamera", typeof(Matrix3D), typeof(Camera), new FrameworkPropertyMetadata(Matrix3D.Identity));
        public Matrix3D TransformCamera
        {
            get { return (Matrix3D)GetValue(TransformCameraProperty); }
        }
        public static DependencyProperty TransformViewProperty = DependencyProperty.Register("TransformView", typeof(Matrix3D), typeof(Camera), new FrameworkPropertyMetadata(Matrix3D.Identity));
        public Matrix3D TransformView
        {
            get { return (Matrix3D)GetValue(TransformViewProperty); }
        }
        public static DependencyProperty TransformProjectionProperty = DependencyProperty.Register("TransformProjection", typeof(Matrix3D), typeof(Camera), new FrameworkPropertyMetadata(Matrix3D.Identity));
        public Matrix3D TransformProjection
        {
            get { return (Matrix3D)GetValue(TransformProjectionProperty); }
        }
        public static DependencyProperty TransformModelViewProjectionProperty = DependencyProperty.Register("TransformModelViewProjection", typeof(Matrix3D), typeof(Camera), new FrameworkPropertyMetadata(Matrix3D.Identity));
        public Matrix3D TransformModelViewProjection
        {
            get { return (Matrix3D)GetValue(TransformModelViewProjectionProperty); }
        }
        public Matrix3D Transform
        {
            set { Location.Transform = value; UpdateTransform(); }
        }
        public Camera()
        {
            UpdateTransform();
        }
        public void RotatePost(Quaternion rotate)
        {
            Location.RotatePost(rotate);
            UpdateTransform();
        }
        public void RotatePre(Quaternion rotate)
        {
            Location.RotatePre(rotate);
            UpdateTransform();
        }
        public void TranslatePost(Vector3D offset)
        {
            Location.TranslatePost(offset);
            UpdateTransform();
        }
        void UpdateTransform()
        {
            SetValue(TransformCameraProperty, Location.Transform);
            SetValue(TransformViewProperty, MathHelp.Invert(Location.Transform));
            SetValue(TransformProjectionProperty, CameraProjection.Projection);
            SetValue(TransformModelViewProjectionProperty, MathHelp.Invert(Location.Transform) * CameraProjection.Projection);
            InvalidateProperty(TransformModelViewProjectionProperty);
        }
        readonly TransformQuaternion Location = new TransformQuaternion(new Vector3D(0, 2, -5));
        readonly Perspective CameraProjection = new Perspective();
    }
}