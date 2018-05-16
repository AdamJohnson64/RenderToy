////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.Transforms;
using RenderToy.Utility;
using System.Windows;

namespace RenderToy.WPF
{
    public class Camera : DependencyObject
    {
        public static DependencyProperty ModelViewProjectionProperty = DependencyProperty.Register("ModelViewProjection", typeof(Matrix3D), typeof(Camera));
        public Matrix3D ModelViewProjection
        {
            get { return (Matrix3D)GetValue(ModelViewProjectionProperty); }
            set { SetValue(ModelViewProjectionProperty, value); }
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
            SetValue(ModelViewProjectionProperty, MathHelp.Invert(Location.Transform) * Projection.Projection);
            InvalidateProperty(ModelViewProjectionProperty);
        }
        readonly TransformQuaternion Location = new TransformQuaternion(new Vector3D(0, 2, -5));
        readonly Perspective Projection = new Perspective();
    }
}