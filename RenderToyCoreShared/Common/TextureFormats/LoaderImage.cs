////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.TextureFormats;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace RenderToy.Textures
{
    class LoaderImage
    {
        public static ImageBgra32 LoadFromPath(string path)
        {
            ImageBgra32 result = null;
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
        public static ConfiguredTaskAwaitable<ImageBgra32> LoadFromPathAsync(string path)
        {
            return Task.Run(() =>
            {
                return LoadFromPath(path);
            }).ConfigureAwait(false);
        }
    }
}