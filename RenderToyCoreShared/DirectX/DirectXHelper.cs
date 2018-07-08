////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Expressions;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.PipelineModel;
using RenderToy.Primitives;
using RenderToy.Textures;
using System;
using System.Linq;

namespace RenderToy.DirectX
{
    public struct DirectXHelper
    {
        public static float[] ConvertToD3DMatrix(Matrix3D matrix)
        {
            return new float[16]
            {
                (float)matrix.M11, (float)matrix.M12, (float)matrix.M13, (float)matrix.M14,
                (float)matrix.M21, (float)matrix.M22, (float)matrix.M23, (float)matrix.M24,
                (float)matrix.M31, (float)matrix.M32, (float)matrix.M33, (float)matrix.M34,
                (float)matrix.M41, (float)matrix.M42, (float)matrix.M43, (float)matrix.M44,
            };
        }
        public static XYZ[] ConvertToXYZ(IPrimitive primitive)
        {
            var verticesin = PrimitiveAssembly.CreateTrianglesDX(primitive);
            var verticesout = verticesin.Select(i => new XYZ
            {
                Position = new Vector3F((float)i.Position.X, (float)i.Position.Y, (float)i.Position.Z)
            });
            return verticesout.ToArray();
        }
        public static XYZNorDiffuseTex1[] ConvertToXYZNorDiffuseTex1(IPrimitive primitive)
        {
            var verticesin = PrimitiveAssembly.CreateTrianglesDX(primitive);
            var verticesout = verticesin.Select(i => new XYZNorDiffuseTex1
            {
                Position = new Vector3F((float)i.Position.X, (float)i.Position.Y, (float)i.Position.Z),
                Normal = new Vector3F((float)i.Normal.X, (float)i.Normal.Y, (float)i.Normal.Z),
                Diffuse = i.Diffuse,
                TexCoord = new Vector2F((float)i.TexCoord.X, (float)i.TexCoord.Y),
                Tangent = new Vector3F((float)i.Tangent.X, (float)i.Tangent.Y, (float)i.Tangent.Z),
                Bitangent = new Vector3F((float)i.Bitangent.X, (float)i.Bitangent.Y, (float)i.Bitangent.Z),
            });
            return verticesout.ToArray();
        }
        public static void ConvertToBitmap(IImageBgra32 node, IntPtr bitmapptr, int bitmapWidth, int bitmapHeight, int bitmapstride)
        {
            if (node == null || bitmapptr == IntPtr.Zero) return;
            unsafe
            {
                for (int y = 0; y < bitmapHeight; ++y)
                {
                    void* raster = (byte*)bitmapptr.ToPointer() + bitmapstride * y;
                    for (int x = 0; x < bitmapWidth; ++x)
                    {
                        ((uint*)raster)[x] = node.GetImagePixel(x, y);
                    }
                }
            }
        }
        public static IImageBgra32 GetImageConverter(IMaterial node, int suggestedWidth, int suggestedHeight)
        {
            if (node == null) return GetImageConverter(StockMaterials.Missing, ThumbnailSize, ThumbnailSize);
            System.Type type = node.GetType();
            if (node is ITexture)
            {
                return GetImageConverter(((ITexture)node).GetTextureLevel(0), suggestedWidth, suggestedHeight);
            }
            else if (node is IImageBgra32)
            {
                return (IImageBgra32)node;
            }
            else if (node is IMNNode<double>)
            {
                var lambda = ((IMNNode<double>)node).CompileMSIL();
                var context = new EvalContext();
                return new ImageConverterAdaptor(suggestedWidth, suggestedHeight, (x, y) =>
                {
                    context.U = (x + 0.5) / suggestedWidth;
                    context.V = (y + 0.5) / suggestedHeight;
                    double v = lambda(context);
                    return Rasterization.ColorToUInt32(new Vector4D(v, v, v, 1));
                });
            }
            else if (node is IMNNode<Vector4D>)
            {
                var lambda = ((IMNNode<Vector4D>)node).CompileMSIL();
                var context = new EvalContext();
                return new ImageConverterAdaptor(suggestedWidth, suggestedHeight, (x, y) =>
                {
                    context.U = (x + 0.5) / suggestedWidth;
                    context.V = (y + 0.5) / suggestedHeight;
                    return Rasterization.ColorToUInt32(lambda(context));
                });
            }
            else
            {
                return GetImageConverter(StockMaterials.Missing, ThumbnailSize, ThumbnailSize);
            }
        }
        class ImageConverterAdaptor : IImageBgra32
        {
            public ImageConverterAdaptor(int width, int height, Func<int, int, uint> sampler)
            {
                Width = width;
                Height = height;
                Sampler = sampler;
            }
            public bool IsConstant() { return false; }
            public int GetImageWidth() { return Width; }
            public int GetImageHeight() { return Height; }
            public uint GetImagePixel(int x, int y) { return Sampler(x, y); }
            int Width, Height;
            Func<int, int, uint> Sampler;
        }
        public static readonly int ThumbnailSize = 32;
    }
}