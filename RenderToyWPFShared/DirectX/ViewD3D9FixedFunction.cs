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
            var constants = new Dictionary<string, object>();
            constants["transformAspect"] = Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            constants["transformCamera"] = AttachedView.GetTransformCamera(this);
            constants["transformView"] = AttachedView.GetTransformView(this);
            constants["transformProjection"] = AttachedView.GetTransformProjection(this);
            Direct3D9Helper.CreateSceneDrawFixedFunction(AttachedView.GetScene(this))(constants);
        }
    }
}