////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.SceneGraph;
using System.Collections.Generic;

namespace RenderToy.RenderControl
{
    struct BitmapBuffer
    {
        public BitmapBuffer(IEnumerable<TransformedObject> scene, Matrix3D mvp, int width, int height, byte[] buffer)
        {
            Scene = scene;
            MVP = mvp;
            Width = width;
            Height = height;
            Buffer = buffer;
        }
        public IEnumerable<TransformedObject> Scene;
        public Matrix3D MVP;
        public int Width;
        public int Height;
        public byte[] Buffer;
    }
}