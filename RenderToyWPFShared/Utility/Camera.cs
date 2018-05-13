////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Transforms;
using RenderToy.Utility;
using System.Windows;

namespace RenderToy.WPF
{
    public class Camera : DependencyObject
    {
        public static DependencyProperty TransformProperty = DependencyProperty.Register("Transform", typeof(Matrix3D), typeof(Camera));
        public Camera()
        {
            Object = new TransformQuaternion(new Vector3D(0, 2, -5));
            SetValue(TransformProperty, Object.Transform);
            Object.OnTransformChanged += () => SetValue(TransformProperty, Object.Transform);
        }
        public readonly TransformQuaternion Object;
    }
}