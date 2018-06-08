////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;

namespace RenderToy.RenderControl
{
    /// <summary>
    /// Degenerate multipass handler that performs all work in a single pass.
    /// This model will be used for renderers that converge quickly or don't
    /// require any fancy sampling (i.e. the raycase, TBN or raytrace).
    /// This version runs synchronously and will block on CopyTo().
    /// </summary>
    class SinglePassSyncAdaptor : IMultiPass
    {
        public SinglePassSyncAdaptor(RenderCall fillwith, BitmapReady onbitmapready)
        {
            this.fillwith = fillwith;
            this.onbitmapready = onbitmapready;
        }
        public override string ToString() { return RenderCall.GetDisplayNameFull(fillwith.MethodInfo.Name); }
        public void SetScene(IScene scene) { this.scene = scene; InvalidateFrame(); }
        public void SetCamera(Matrix3D mvp)
        {
            if (this.mvp == mvp) return;
            this.mvp = mvp;
            InvalidateFrame();
        }
        public void SetTarget(int width, int height)
        {
            if (this.width == width && this.height == height) return;
            this.width = width;
            this.height = height;
            InvalidateFrame();
        }
        public void CopyTo(IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            fillwith.Action(scene, mvp, bitmap_ptr, render_width, render_height, bitmap_stride, null);
        }
        void InvalidateFrame()
        {
            if (onbitmapready != null) onbitmapready();
        }
        RenderCall fillwith;
        IScene scene;
        Matrix3D mvp;
        int width, height;
        BitmapReady onbitmapready;
    }
}