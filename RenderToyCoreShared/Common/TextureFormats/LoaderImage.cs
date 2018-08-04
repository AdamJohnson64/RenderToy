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
            if (path.ToUpperInvariant().EndsWith(".HDR"))
            {
                return LoaderHDR.LoadFromPath(path);
            }
            else if (path.ToUpperInvariant().EndsWith(".PNG"))
            {
                return LoaderPNG.LoadFromPath(path);
            }
            else if (path.ToUpperInvariant().EndsWith(".TGA"))
            {
                return LoaderTGA.LoadFromPath(path);
            }
            else
            {
                return null;
            }
        }
    }
}