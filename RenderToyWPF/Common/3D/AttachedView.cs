﻿using RenderToy.DocumentModel;
using RenderToy.Math;
using System.Windows;

namespace RenderToy.WPF
{
    public static class AttachedView
    {
        #region - Section : Dependency Properties -
        public static DependencyProperty TransformCameraProperty = DependencyProperty.RegisterAttached("TransformCamera", typeof(Matrix3D), typeof(AttachedView), new FrameworkPropertyMetadata(Matrix3D.Identity));
        public static Matrix3D GetTransformCamera(DependencyObject from)
        {
            return (Matrix3D)from.GetValue(TransformCameraProperty);
        }
        public static void SetTransformCamera(DependencyObject on, Matrix3D value)
        {
            on.SetValue(TransformCameraProperty, value);
        }
        public static DependencyProperty TransformViewProperty = DependencyProperty.RegisterAttached("TransformView", typeof(Matrix3D), typeof(AttachedView), new FrameworkPropertyMetadata(Matrix3D.Identity));
        public static Matrix3D GetTransformView(DependencyObject from)
        {
            return (Matrix3D)from.GetValue(TransformViewProperty);
        }
        public static void SetTransformView(DependencyObject on, Matrix3D value)
        {
            on.SetValue(TransformViewProperty, value);
        }
        public static DependencyProperty TransformProjectionProperty = DependencyProperty.RegisterAttached("TransformProjection", typeof(Matrix3D), typeof(AttachedView), new FrameworkPropertyMetadata(Matrix3D.Identity));
        public static Matrix3D GetTransformProjection(DependencyObject from)
        {
            return (Matrix3D)from.GetValue(TransformProjectionProperty);
        }
        public static void SetTransformProjection(DependencyObject on, Matrix3D value)
        {
            on.SetValue(TransformProjectionProperty, value);
        }
        public static DependencyProperty SceneProperty = DependencyProperty.RegisterAttached("Scene", typeof(SparseScene), typeof(AttachedView));
        public static SparseScene GetScene(DependencyObject from)
        {
            return (SparseScene)from.GetValue(SceneProperty);
        }
        public static void SetScene(DependencyObject on, SparseScene value)
        {
            on.SetValue(SceneProperty, value);
        }
        #endregion
    }
}