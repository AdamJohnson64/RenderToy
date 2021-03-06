﻿using RenderToyCOM;
using RenderToy.Textures;
using System.IO;

namespace RenderToy.TextureFormats
{
    public static class LoaderTGA
    {
        public static Surface LoadFromPath(string path)
        {
            if (!File.Exists(path)) return null;
            using (var streamreader = File.OpenRead(path))
            {
                var binaryreader = new BinaryReader(streamreader);
                byte imageid = binaryreader.ReadByte();
                if (imageid != 0) throw new FileLoadException("Expected Image ID 0 (Extensions not supported).");
                byte colormaptype = binaryreader.ReadByte();
                if (colormaptype != 0) throw new FileLoadException("Expected Color Map Type 0 (No palette).");
                byte imagetype = binaryreader.ReadByte();
                if (imagetype != 2) throw new FileLoadException("Expected Image Type 2 (Uncompressed True-color).");
                ushort colormapfirst = binaryreader.ReadUInt16();
                if (colormapfirst != 0) throw new FileLoadException("Expected Zero Colormap.");
                ushort colormapcount = binaryreader.ReadUInt16();
                if (colormapcount != 0) throw new FileLoadException("Expected Zero Colormap.");
                byte colormapentrysize = binaryreader.ReadByte();
                if (colormapentrysize != 0) throw new FileLoadException("Expected Zero Colormap.");
                ushort xorigin = binaryreader.ReadUInt16();
                ushort yorigin = binaryreader.ReadUInt16();
                ushort width = binaryreader.ReadUInt16();
                ushort height = binaryreader.ReadUInt16();
                byte bitdepth = binaryreader.ReadByte();
                byte imagedescriptor = binaryreader.ReadByte();
                byte[] data;
                if (bitdepth == 24)
                {
                    if (imagedescriptor != 0) throw new FileLoadException("Expected Zero Image Descriptor.");
                    data = new byte[4 * width * height];
                    for (int y = height - 1; y >= 0; --y)
                    {
                        for (int x = 0; x < width; ++x)
                        {
                            data[0 + 4 * x + 4 * width * y] = (byte)streamreader.ReadByte();
                            data[1 + 4 * x + 4 * width * y] = (byte)streamreader.ReadByte();
                            data[2 + 4 * x + 4 * width * y] = (byte)streamreader.ReadByte();
                            data[3 + 4 * x + 4 * width * y] = 255;
                        }
                    }
                    return new Surface(Path.GetFileName(path), DXGI_FORMAT.DXGI_FORMAT_B8G8R8X8_UNORM, width, height, data);
                }
                else if (bitdepth == 32)
                {
                    if (imagedescriptor != 8) throw new FileLoadException("Expected Zero Image Descriptor.");
                    data = new byte[4 * width * height];
                    for (int y = height - 1; y >= 0; --y)
                    {
                        for (int x = 0; x < width; ++x)
                        {
                            data[0 + 4 * x + 4 * width * y] = (byte)streamreader.ReadByte();
                            data[1 + 4 * x + 4 * width * y] = (byte)streamreader.ReadByte();
                            data[2 + 4 * x + 4 * width * y] = (byte)streamreader.ReadByte();
                            data[3 + 4 * x + 4 * width * y] = (byte)streamreader.ReadByte();
                        }
                    }
                    return new Surface(Path.GetFileName(path), DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, width, height, data);
                }
                else
                {
                    throw new FileLoadException("Expected 24 or 32bpp.");
                }
            }
        }
    }
}