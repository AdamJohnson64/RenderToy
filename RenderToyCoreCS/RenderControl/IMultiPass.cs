////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using System;

namespace RenderToy.RenderControl
{
    /// <summary>
    /// Contract for renderers which can accumulate output and deliver ARGB bitmaps on request.
    /// </summary>
    public interface IMultiPass
    {
        /// <summary>
        /// Configure the camera for the render.
        /// </summary>
        /// <param name="mvp">The model-view-projection for the camera.</param>
        void SetCamera(Matrix3D mvp);
        /// <summary>
        /// Configure the scene for the render.
        /// </summary>
        /// <param name="scene">The scene graph object.</param>
        void SetScene(Scene scene);
        /// <summary>
        /// Configure the output target dimensions for the render.
        /// </summary>
        /// <param name="width">The pixel width of the target.</param>
        /// <param name="height">The pixel height of the target.</param>
        void SetTarget(int width, int height);
        /// <summary>
        /// Copy the ARGB buffer to a user supplied buffer.
        /// </summary>
        /// <param name="bitmap_ptr">Pointer to the buffer.</param>
        /// <param name="render_width">The pixel width of the buffer.</param>
        /// <param name="render_height">The pixel height of the buffer.</param>
        /// <param name="bitmap_stride">The pitch/stride of the buffer in bytes.</param>
        void CopyTo(IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride);
    }
    public delegate void BitmapReady();
}