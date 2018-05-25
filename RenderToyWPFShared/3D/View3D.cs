////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using RenderToy.Utility;
using System.Windows;

namespace RenderToy.WPF
{
    public static class View3D
    {
        #region - Section : Dependency Properties -
        public static DependencyProperty ModelViewProjectionProperty = DependencyProperty.RegisterAttached("ModelViewProjection", typeof(Matrix3D), typeof(View3D), new FrameworkPropertyMetadata(Matrix3D.Identity, FrameworkPropertyMetadataOptions.AffectsRender));
        public static Matrix3D GetModelViewProjection(DependencyObject from)
        {
            return (Matrix3D)from.GetValue(ModelViewProjectionProperty);
        }
        public static void SetModelViewProjection(DependencyObject on, Matrix3D value)
        {
            on.SetValue(ModelViewProjectionProperty, value);
        }
        public static DependencyProperty SceneProperty = DependencyProperty.RegisterAttached("Scene", typeof(IScene), typeof(View3D), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));
        public static IScene GetScene(DependencyObject from)
        {
            return (IScene)from.GetValue(SceneProperty);
        }
        public static void SetScene(DependencyObject on, IScene value)
        {
            on.SetValue(SceneProperty, value);
        }
        #endregion
    }
}