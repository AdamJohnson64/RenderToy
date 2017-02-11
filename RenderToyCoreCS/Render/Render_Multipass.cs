////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;

namespace RenderToy
{
    /// <summary>
    /// Multipass interface for multipass render modes.
    /// </summary>
    interface IMultipass
    {
        void SetScene(Scene scene);
        void SetCamera(Matrix3D mvp);
        void SetTarget(int width, int height);
        void Start();
        bool CopyTo(IntPtr buffer_ptr, int buffer_width, int buffer_height, int buffer_stride);
    }
    /// <summary>
    /// Degenerate multipass handler that performs all work in a single pass.
    /// This model will be used for renderers that converge quickly or don't
    /// require any fancy sampling (i.e. the raycase, TBN or raytrace).
    /// </summary>
    class SinglePass : IMultipass
    {
        public SinglePass(RenderCall fillwith) { this.fillwith = fillwith; }
        public override string ToString() { return RenderCall.GetDisplayNameFull(fillwith.MethodInfo.Name); }
        public void SetScene(Scene scene) { this.scene = scene; }
        public void SetCamera(Matrix3D mvp) { this.mvp = mvp; }
        public void SetTarget(int width, int height) { }
        public void Start() { }
        public bool CopyTo(IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            fillwith.Action(scene, mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
            return false;
        }
        RenderCall fillwith;
        Scene scene;
        Matrix3D mvp;
    }
}