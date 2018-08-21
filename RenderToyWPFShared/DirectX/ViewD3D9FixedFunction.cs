////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Math;
using System.Collections.Generic;
using System.Windows;

namespace RenderToy.WPF
{
    public class ViewD3D9FixedFunction : ViewD3DImageDirect
    {
        static ViewD3D9FixedFunction()
        {
            AttachedView.SceneProperty.OverrideMetadata(typeof(ViewD3D9FixedFunction), new FrameworkPropertyMetadata(null, (s, e) => ((ViewD3D9FixedFunction)s).InvalidateVisual()));
            AttachedView.TransformViewProperty.OverrideMetadata(typeof(ViewD3D9FixedFunction), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) => ((ViewD3D9FixedFunction)s).InvalidateVisual()));
            AttachedView.TransformProjectionProperty.OverrideMetadata(typeof(ViewD3D9FixedFunction), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) => ((ViewD3D9FixedFunction)s).InvalidateVisual()));
        }
        protected override void RenderD3D()
        {
            var constants = new Dictionary<string, object>();
            constants["transformAspect"] = Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            constants["transformCamera"] = AttachedView.GetTransformCamera(this);
            constants["transformView"] = AttachedView.GetTransformView(this);
            constants["transformProjection"] = AttachedView.GetTransformProjection(this);
            Direct3D9Helper.CreateSceneDrawFixedFunction(AttachedView.GetScene(this))(constants);
        }
    }
}