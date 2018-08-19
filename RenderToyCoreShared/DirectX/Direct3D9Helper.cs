////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.DocumentModel;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.Textures;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace RenderToy.DirectX
{
    public static class Direct3D9Helper
    {
        #region - Section : Direct3D Render Helpers -
        public static Action<Dictionary<string, object>> CreateSceneDrawFixedFunction(SparseScene scene)
        {
            if (scene == null) return (constants) => { };
            return (constants) =>
            {
                var transformAspect = (Matrix3D)constants["transformAspect"];
                var transformView = (Matrix3D)constants["transformView"];
                var transformProjection = (Matrix3D)constants["transformProjection"];
                var transformTail = transformView * transformProjection * transformAspect;
                foreach (var transformedobject in scene)
                {
                    var createdvertexbuffer = Direct3D9Helper.CreateVertexBuffer(transformedobject.NodePrimitive);
                    if (createdvertexbuffer.VertexBuffer == null) continue;
                    device.SetStreamSource(0, createdvertexbuffer.VertexBuffer, 0U, (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)));
                    device.SetTexture(0, Direct3D9Helper.CreateTexture(transformedobject.NodeMaterial, null));
                    device.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(Direct3DHelper.ConvertToD3DMatrix(transformedobject.Transform * transformTail), 0));
                    device.DrawPrimitive(D3DPrimitiveType.TriangleList, 0U, (uint)createdvertexbuffer.PrimitiveCount);
                }
            };
        }
        public static Action<Dictionary<string, object>> CreateSceneDraw(SparseScene scene)
        {
            if (scene == null) return (constants) => { };
            return (constants) =>
            {
                var transformAspect = (Matrix3D)constants["transformAspect"];
                var transformCamera = (Matrix3D)constants["transformCamera"];
                var transformView = (Matrix3D)constants["transformView"];
                var transformProjection = (Matrix3D)constants["transformProjection"];
                var transformTail = transformView * transformProjection * transformAspect;
                foreach (var transformedobject in scene)
                {
                    if (transformedobject.NodePrimitive == null) continue;
                    var transformModel = transformedobject.Transform;
                    var transformModelViewProjection = transformModel * transformTail;
                    var createdvertexbuffer = Direct3D9Helper.CreateVertexBuffer(transformedobject.NodePrimitive);
                    if (createdvertexbuffer.VertexBuffer == null) continue;
                    Direct3D9Helper.device.SetStreamSource(0, createdvertexbuffer.VertexBuffer, 0U, (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)));
                    var objmat = transformedobject.NodeMaterial as OBJMaterial;
                    Direct3D9Helper.device.SetTexture(0, Direct3D9Helper.CreateTexture(objmat == null ? transformedobject.NodeMaterial : objmat.map_Kd, StockMaterials.PlasticWhite));
                    Direct3D9Helper.device.SetTexture(1, Direct3D9Helper.CreateTexture(objmat == null ? null : objmat.map_d, StockMaterials.PlasticWhite));
                    Direct3D9Helper.device.SetTexture(2, Direct3D9Helper.CreateTexture(objmat == null ? null : objmat.map_bump, StockMaterials.PlasticLightBlue));
                    Direct3D9Helper.device.SetTexture(3, Direct3D9Helper.CreateTexture(objmat == null ? null : objmat.displacement, StockMaterials.PlasticWhite));
                    Direct3D9Helper.device.SetVertexShaderConstantF(0, Marshal.UnsafeAddrOfPinnedArrayElement(Direct3DHelper.ConvertToD3DMatrix(transformCamera), 0), 4);
                    Direct3D9Helper.device.SetVertexShaderConstantF(4, Marshal.UnsafeAddrOfPinnedArrayElement(Direct3DHelper.ConvertToD3DMatrix(transformModel), 0), 4);
                    Direct3D9Helper.device.SetVertexShaderConstantF(8, Marshal.UnsafeAddrOfPinnedArrayElement(Direct3DHelper.ConvertToD3DMatrix(transformView), 0), 4);
                    Direct3D9Helper.device.SetVertexShaderConstantF(12, Marshal.UnsafeAddrOfPinnedArrayElement(Direct3DHelper.ConvertToD3DMatrix(transformProjection), 0), 4);
                    Direct3D9Helper.device.SetVertexShaderConstantF(16, Marshal.UnsafeAddrOfPinnedArrayElement(Direct3DHelper.ConvertToD3DMatrix(transformModelViewProjection), 0), 4);
                    Direct3D9Helper.device.DrawPrimitive(D3DPrimitiveType.TriangleList, 0U, (uint)createdvertexbuffer.PrimitiveCount);
                }
            };
        }
        #endregion
        #region - Section : Direct3D Resource Factory -
        public static Direct3DTexture9 CreateTexture(IMaterial material, IMaterial missing)
        {
            if (material == null) material = missing;
            if (material == null) material = StockMaterials.Missing;
            return MementoServer.Default.Get(material, GeneratedTextureToken, () =>
            {
                var astexture = material as ITexture;
                if (astexture != null)
                {
                    var level0 = astexture.GetSurface(0, 0);
                    if (level0 == null) return null;
                    var texture = Direct3D9Helper.device.CreateTexture((uint)level0.GetImageWidth(), (uint)level0.GetImageHeight(), (uint)astexture.GetTextureLevelCount(), 0U, D3DFormat.A8R8G8B8, D3DPool.Default, null);
                    var texturescratch = Direct3D9Helper.device.CreateTexture((uint)level0.GetImageWidth(), (uint)level0.GetImageHeight(), (uint)astexture.GetTextureLevelCount(), 0U, D3DFormat.A8R8G8B8, D3DPool.SystemMemory, null);
                    for (int level = 0; level < astexture.GetTextureLevelCount(); ++level)
                    {
                        D3DLockedRect lockit = texturescratch.LockRect((uint)level);
                        var thislevel = astexture.GetSurface(0, level);
                        thislevel.ConvertToBitmap(lockit.Bits, thislevel.GetImageWidth(), thislevel.GetImageHeight(), lockit.Pitch);
                        texturescratch.UnlockRect((uint)level);
                    }
                    Direct3D9Helper.device.UpdateTexture(texturescratch, texture);
                    return texture;
                }
                else
                {
                    var asimage = material.GetImageConverter(512, 512);
                    var texture = Direct3D9Helper.device.CreateTexture((uint)asimage.GetImageWidth(), (uint)asimage.GetImageHeight(), 1, 0U, D3DFormat.A8R8G8B8, D3DPool.Default, null);
                    var texturescratch = Direct3D9Helper.device.CreateTexture((uint)asimage.GetImageWidth(), (uint)asimage.GetImageHeight(), 1, 0U, D3DFormat.A8R8G8B8, D3DPool.SystemMemory, null);
                    D3DLockedRect lockit = texturescratch.LockRect(0);
                    asimage.ConvertToBitmap(lockit.Bits, asimage.GetImageWidth(), asimage.GetImageHeight(), lockit.Pitch);
                    texturescratch.UnlockRect(0);
                    Direct3D9Helper.device.UpdateTexture(texturescratch, texture);
                    return texture;
                }
            });
        }
        public static VertexBufferInfo CreateVertexBuffer(IPrimitive primitive)
        {
            if (primitive == null) return new VertexBufferInfo { PrimitiveCount = 0, VertexBuffer = null };
            return MementoServer.Default.Get(primitive, GeneratedVertexBufferToken, () =>
            {
                var data = Direct3DHelper.ConvertToXYZNorDiffuseTex1(primitive);
                var size = (uint)(Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) * data.Length);
                VertexBufferInfo buffer = new VertexBufferInfo();
                if (data.Length > 0)
                {
                    buffer.VertexBuffer = Direct3D9Helper.device.CreateVertexBuffer(size, 0, 0U, D3DPool.Default, null);
                    var locked = buffer.VertexBuffer.Lock(0U, size, 0U);
                    unsafe
                    {
                        Buffer.MemoryCopy(Marshal.UnsafeAddrOfPinnedArrayElement(data, 0).ToPointer(), locked.ToPointer(), size, size);
                    }
                    buffer.VertexBuffer.Unlock();
                }
                buffer.PrimitiveCount = data.Length / 3;
                return buffer;
            });
        }
        public struct VertexBufferInfo
        {
            public Direct3DVertexBuffer9 VertexBuffer;
            public int PrimitiveCount;
        }
        static readonly string GeneratedTextureToken = "DirectX9Texture";
        static readonly string GeneratedVertexBufferToken = "DirectX9VertexBuffer";
        #endregion
        #region - Section : Direct3D Global Objects -
        static readonly Direct3D9Ex d3d = new Direct3D9Ex();
        static readonly Form form = new Form();
        public static readonly Direct3DDevice9Ex device = d3d.CreateDevice(form.Handle);
        public static Direct3DVertexDeclaration9 vertexdeclaration = device.CreateVertexDeclaration(new D3DVertexElement9[] {
            new D3DVertexElement9 { Stream = 0, Offset = 0, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Position, UsageIndex = 0 },
            new D3DVertexElement9 { Stream = 0, Offset = 12, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Normal, UsageIndex = 0 },
            new D3DVertexElement9 { Stream = 0, Offset = 24, Type = D3DDeclType.D3DColor, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Color, UsageIndex = 0 },
            new D3DVertexElement9 { Stream = 0, Offset = 28, Type = D3DDeclType.Float2, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.TexCoord, UsageIndex = 0 },
            new D3DVertexElement9 { Stream = 0, Offset = 36, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Tangent, UsageIndex = 0 },
            new D3DVertexElement9 { Stream = 0, Offset = 48, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Binormal, UsageIndex = 0 },
        });
        #endregion
    }
}