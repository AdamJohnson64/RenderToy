////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System.Windows;

namespace RenderToy.Textures
{
    class SurfaceCrossToCube : ITexture
    {
        public SurfaceCrossToCube(Surface source)
        {
            Source = source;
        }
        public ISurface GetSurface(int array, int level)
        {
            int w = Source.GetImageWidth();
            int h = Source.GetImageHeight();
            var facenames = new[] { "-X", "+X", "-Y", "+Y", "-Z", "+Z" };
            var facerects = new[]
            {
                new Int32Rect { X = w * 0 / 3, Y = h * 1 / 4, Width = w / 3, Height = h / 4 },
                new Int32Rect { X = w * 2 / 3, Y = h * 1 / 4, Width = w / 3, Height = h / 4 },
                new Int32Rect { X = w * 1 / 3, Y = h * 0 / 4, Width = w / 3, Height = h / 4 },
                new Int32Rect { X = w * 1 / 3, Y = h * 2 / 4, Width = w / 3, Height = h / 4 },
                new Int32Rect { X = w * 1 / 3, Y = h * 3 / 4, Width = w / 3, Height = h / 4 },
                new Int32Rect { X = w * 1 / 3, Y = h * 1 / 4, Width = w / 3, Height = h / 4 },
            };
            var rect = facerects[array];
            return new SurfaceRegion(Source, rect.X, rect.Y, rect.Width, rect.Height);
        }
        public int GetTextureArrayCount()
        {
            return 6;
        }
        public int GetTextureLevelCount()
        {
            return 1;
        }
        public bool IsConstant()
        {
            return false;
        }
        readonly Surface Source;
    }
}