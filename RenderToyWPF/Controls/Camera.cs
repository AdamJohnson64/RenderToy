////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Windows;

namespace RenderToy
{
    public class Camera : DependencyObject
    {
        public static DependencyProperty TransformProperty = DependencyProperty.Register("Transform", typeof(Matrix3D), typeof(Camera));
        public Camera()
        {
            Object = new TransformPosQuat(new Vector3D(0, 2, -5));
            SetValue(TransformProperty, Object.Transform);
            Object.OnTransformChanged += () => SetValue(TransformProperty, Object.Transform);
        }
        public readonly TransformPosQuat Object;
    }
}