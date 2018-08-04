////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Textures;

namespace RenderToy.TextureFormats
{
    public static class LoaderPNG
    {
        public static ImageBgra32 LoadFromPath(string path)
        {
            var image = LibPNG.Open(path);
            return new ImageBgra32(path, image.Width, image.Height, image.Data);
        }
    }
}