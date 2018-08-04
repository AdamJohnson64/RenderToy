////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Textures;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace RenderToy.TextureFormats
{
    public static class LoaderHDR
    {
        public static ImageBgra32 LoadFromPath(string path)
        {
            if (!File.Exists(path)) return null;
            string imageFormat = null;
            float imageExposure = 1;
            int imageWidth = 0;
            int imageHeight = 0;
            using (var binaryreader = new BinaryReader(File.OpenRead(path)))
            {
                {
                    Func<string> ReadLine = () =>
                    {
                        var bytes = new List<byte>();
                        byte readchar = binaryreader.ReadByte();
                        while (readchar != 0x0A)
                        {
                            bytes.Add(readchar);
                            readchar = binaryreader.ReadByte();
                        }
                        return Encoding.ASCII.GetString(bytes.ToArray());
                    };
                    var line = ReadLine();
                    // Expect the magic file string.
                    if (line != "#?RADIANCE") throw new FileLoadException("The file '" + Path.GetFileName(path) + "' is not a HDR file.");
                    // Keep reading the text header until we encounter a blank line.
                    while (!string.IsNullOrWhiteSpace(line))
                    {
                        line = ReadLine();
                        if (line.StartsWith("#")) continue;
                        if (line.StartsWith("FORMAT"))
                        {
                            int equalsat = line.IndexOf('=');
                            imageFormat = line.Substring(equalsat + 1).Trim();
                            continue;
                        }
                        if (line.StartsWith("EXPOSURE"))
                        {
                            int equalsat = line.IndexOf('=');
                            var value = line.Substring(equalsat + 1).Trim();
                            imageExposure = float.Parse(value);
                            continue;
                        }
                    }
                    line = ReadLine();
                    var match = Regex.Match(line, "-Y (?'height'[0-9]*) \\+X (?'width'[0-9]*)");
                    if (match == null || !match.Success) throw new Exception("The file '" + Path.GetFileName(path) + "' has a bad size definition '" + line + "'.");
                    imageWidth = int.Parse(match.Groups["width"].Value);
                    imageHeight = int.Parse(match.Groups["height"].Value);
                }
                byte[] imageData = new byte[4 * imageWidth * imageHeight];
                for (int y = 0; y < imageHeight; ++y)
                {
                    byte[] rasterBuffer = new byte[4 * imageWidth];
                    // Read the length of this raster (should be equal to the image width).
                    ushort length = 0;
                    {
                        byte readbyte = binaryreader.ReadByte();
                        if (readbyte != 2) throw new Exception("HDR scanline is invalid.");
                        readbyte = binaryreader.ReadByte();
                        if (readbyte != 2) throw new Exception("HDR scanline is invalid.");
                    }
                    length |= (ushort)(binaryreader.ReadByte() << 8);
                    length |= (ushort)(binaryreader.ReadByte() << 0);
                    if (length != imageWidth) throw new Exception("HDR RLE raster has wrong size.");
                    // Read each of the RGBE components in planar layout.
                    for (int currentComponent = 0; currentComponent < 4; ++currentComponent)
                    {
                        int currentPixel = 0;
                        // Keep reading until we reach the end of this raster.
                        while (currentPixel < length)
                        {
                            byte compressionModeAndSize = binaryreader.ReadByte();
                            // A zero length span is meaningless.
                            if (compressionModeAndSize == 0) throw new Exception("HDR RLE has zero length.");
                            // There are two possible encodings for this span:
                            if (compressionModeAndSize > 128)
                            {
                                // Read a replicated run-length encoded span.
                                compressionModeAndSize &= 127;
                                byte valueRLE = binaryreader.ReadByte();
                                for (int replicateComponentRLE = 0; replicateComponentRLE < compressionModeAndSize; ++replicateComponentRLE)
                                {
                                    rasterBuffer[4 * currentPixel + currentComponent] = valueRLE;
                                    ++currentPixel;
                                }
                                // If we ran off the end of the image then something is wrong.
                                if (currentPixel > length) throw new Exception("HDR RLE raster overrun.");
                            }
                            else
                            {
                                // Read a span of bytes, one byte at a time.
                                for (int readAllComponent = 0; readAllComponent < compressionModeAndSize; ++readAllComponent)
                                {
                                    rasterBuffer[4 * currentPixel + currentComponent] = binaryreader.ReadByte();
                                    ++currentPixel;
                                }
                                // If we ran off the end of the image then something is wrong.
                                if (currentPixel > length) throw new Exception("HDR RLE raster overrun.");
                            }
                        }
                    }
                    // Unpack the RGBE encoded pixels into RGBA32 format.
                    for (int x = 0; x < imageWidth; ++x)
                    {
                        byte unpackR = rasterBuffer[x * 4 + 0];
                        byte unpackG = rasterBuffer[x * 4 + 1];
                        byte unpackB = rasterBuffer[x * 4 + 2];
                        byte unpackE = rasterBuffer[x * 4 + 3];
                        imageData[4 * (x + imageWidth * y) + 2] = (byte)(Clamp(RGBEComponentToFloat(unpackR, unpackE) * 255, 0, 255));
                        imageData[4 * (x + imageWidth * y) + 1] = (byte)(Clamp(RGBEComponentToFloat(unpackG, unpackE) * 255, 0, 255));
                        imageData[4 * (x + imageWidth * y) + 0] = (byte)(Clamp(RGBEComponentToFloat(unpackB, unpackE) * 255, 0, 255));
                        imageData[4 * (x + imageWidth * y) + 3] = 0xFF;
                    }
                }
                return new ImageBgra32(path, imageWidth, imageHeight, imageData);
            }
        }
        static float Clamp(float value, float min, float max)
        {
            return value < min ? min : (value < max ? value : max);
        }
        static float RGBEComponentToFloat(int mantissa, int exponent)
        {
            float v = mantissa / 256.0f;
            float d = (float)System.Math.Pow(2, exponent - 128);
            return v * d;
            //return UInt32ToFloat(((uint)mantissa << 0) | ((uint)exponent << 23));
        }
        static float UInt32ToFloat(uint value)
        {
            unsafe
            {
                return *(float*)&value;
            }
        }
    }
}