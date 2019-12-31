using RenderToyCOM;
using RenderToy.Textures;

namespace RenderToy.TextureFormats
{
    public static class LoaderPNG
    {
        public static Surface LoadFromPath(string path)
        {
            var image = LibPNG.Open(path);
            return new Surface(path, DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, image.Width, image.Height, image.Data);
        }
    }
}