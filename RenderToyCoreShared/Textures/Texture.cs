////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Utility;
using System.Collections.Generic;

namespace RenderToy.Textures
{
    public interface ITexture
    {
        int GetTextureLevelCount();
        IImageBgra32 GetTextureLevel(int level);
    }
    class Texture : IMaterial, ITexture, INamed
    {
        public Texture(string name, ImageBgra32 level0, bool generateMips)
        {
            Name = name;
            var levels = new List<ImageBgra32>();
            levels.Add(level0);
            if (generateMips)
            {
                while (true)
                {
                    var lastlevel = levels[levels.Count - 1];
                    var newlevel = lastlevel.BoxFilter();
                    if (newlevel == null) break;
                    levels.Add(newlevel);
                }
            }
            Levels = levels.ToArray();
        }
        public string GetName()
        {
            return Name;
        }
        public bool IsConstant()
        {
            return false;
        }
        public int GetTextureLevelCount()
        {
            return Levels.Length;
        }
        public IImageBgra32 GetTextureLevel(int level)
        {
            return Levels[level];
        }
        string Name;
        ImageBgra32[] Levels;
    }
}