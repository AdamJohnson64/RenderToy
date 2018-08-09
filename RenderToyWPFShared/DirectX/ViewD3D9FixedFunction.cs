////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.DirectX;
using System.Collections.Generic;

namespace RenderToy.WPF
{
    public class ViewD3D9FixedFunction : ViewD3DImageDirect
    {
        protected override void RenderD3D()
        {
            var transformCamera = AttachedView.GetTransformCamera(this);
            var transformView = AttachedView.GetTransformView(this);
            var transformProjection = AttachedView.GetTransformProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            var transformViewProjection = transformView * transformProjection;
            var constants = new Dictionary<string, object>();
            constants["transformCamera"] = transformCamera;
            constants["transformView"] = transformView;
            constants["transformProjection"] = transformProjection;
            constants["transformViewProjection"] = transformViewProjection;
            Direct3D9Helper.CreateSceneDrawFixedFunction(AttachedView.GetScene(this))(constants);
        }
    }
}