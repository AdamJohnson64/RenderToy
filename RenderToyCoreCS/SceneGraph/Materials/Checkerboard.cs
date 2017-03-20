////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy.SceneGraph.Materials
{
    public class Checkerboard : IMaterial
    {
        public Checkerboard(Vector4D color1, Vector4D color2)
        {
            Color1 = color1;
            Color2 = color2;
        }
        private readonly Vector4D Color1;
        private readonly Vector4D Color2;
    }
}