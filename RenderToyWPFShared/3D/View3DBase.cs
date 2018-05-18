////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using RenderToy.Utility;
using System.Windows;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public abstract class View3DBase : FrameworkElement
    {
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
        public static DependencyProperty ModelViewProjectionProperty = DependencyProperty.Register("ModelViewProjection", typeof(Matrix3D), typeof(View3DBase), new FrameworkPropertyMetadata(Matrix3D.Identity, FrameworkPropertyMetadataOptions.AffectsRender));
        public Matrix3D ModelViewProjection
        {
            get { return (Matrix3D)GetValue(ModelViewProjectionProperty); }
            set { SetValue(ModelViewProjectionProperty, value); }
        }
        #endregion
        #region - Section : Rendering -
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            OnRenderToy(drawingContext);
        }
        protected abstract void OnRenderToy(DrawingContext drawingContext);
        #endregion
    }
}