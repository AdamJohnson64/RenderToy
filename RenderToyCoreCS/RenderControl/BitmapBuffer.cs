////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;

namespace RenderToy.RenderControl
{
    struct BitmapBuffer
    {
        public BitmapBuffer(Scene scene, Matrix3D mvp, int width, int height, byte[] buffer)
        {
            Scene = scene;
            MVP = mvp;
            Width = width;
            Height = height;
            Buffer = buffer;
        }
        public Scene Scene;
        public Matrix3D MVP;
        public int Width;
        public int Height;
        public byte[] Buffer;
    }
}