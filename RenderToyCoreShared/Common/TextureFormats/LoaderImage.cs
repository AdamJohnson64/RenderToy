////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.TextureFormats;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace RenderToy.Textures
{
    class LoaderImage
    {
        public static Surface LoadFromPath(string path)
        {
            Surface result = null;
            if (path.ToUpperInvariant().EndsWith(".HDR"))
            {
                result = LoaderHDR.LoadFromPath(path);
            }
            else if (path.ToUpperInvariant().EndsWith(".PNG"))
            {
                result = LoaderPNG.LoadFromPath(path);
            }
            else if (path.ToUpperInvariant().EndsWith(".TGA"))
            {
                result = LoaderTGA.LoadFromPath(path);
            }
            return result;
        }
        public static ConfiguredTaskAwaitable<Surface> LoadFromPathAsync(string path)
        {
            return Task.Run(() =>
            {
                return LoadFromPath(path);
            }).ConfigureAwait(false);
        }
    }
}