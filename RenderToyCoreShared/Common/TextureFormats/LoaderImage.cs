////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.TextureFormats;

namespace RenderToy.Textures
{
    class LoaderImage
    {
        public static ImageBgra32 LoadFromPath(string path)
        {
            if (path.ToUpperInvariant().EndsWith(".TGA"))
            {
                return LoaderTGA.LoadFromPath(path);
            }
            return null;
        }
    }
}