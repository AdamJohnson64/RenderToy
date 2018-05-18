////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Textures;
using System;
using System.IO;

namespace RenderToy.TextureFormats
{
    static class LoaderTGA
    {
        public static Texture24 LoadFromPath(string path)
        {
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
                if (bitdepth != 24) throw new FileLoadException("Expected 24bpp.");
                byte imagedescriptor = binaryreader.ReadByte();
                if (imagedescriptor != 0) throw new FileLoadException("Expected Zero Image Descriptor.");
                var data = new byte[3 * width * height];
                for (int y = 0; y < height; ++y)
                {
                    var raster = binaryreader.ReadBytes(3 * width);
                    Array.Copy(raster, 0, data, 3 * width * (height - y - 1), 3 * width);
                }
                return new Texture24(Path.GetFileName(path), width, height, data);
            }
        }
    }
}