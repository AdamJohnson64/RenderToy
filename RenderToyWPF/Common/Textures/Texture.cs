using RenderToy.Materials;
using RenderToy.Utility;
using System.Collections.Generic;

namespace RenderToy.Textures
{
    public interface ITexture : IMaterial
    {
        int GetTextureArrayCount();
        int GetTextureLevelCount();
        ISurface GetSurface(int array, int level);
    }
    class Texture : ITexture, INamed
    {
        public static Texture Create(string name, Surface level0, bool generateMips)
        {
            return level0 == null ? null : new Texture(name, level0, generateMips);
        }
        public string Name
        {
            get
            {
                return name;
            }
        }
        public bool IsConstant()
        {
            return false;
        }
        public int GetTextureArrayCount()
        {
            return 1;
        }
        public int GetTextureLevelCount()
        {
            return Levels.Length;
        }
        public ISurface GetSurface(int array, int level)
        {
            return Levels[level];
        }
        private Texture(string name, Surface level0, bool generateMips)
        {
            this.name = name;
            var levels = new List<Surface>();
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
        string name;
        Surface[] Levels;
    }
}