////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.Textures;
using RenderToy.Utility;
using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace RenderToy.DirectX
{
    public static class Direct3D9Helper
    {
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
                        DirectXHelper.ConvertToBitmap(thislevel, lockit.Bits, thislevel.GetImageWidth(), thislevel.GetImageHeight(), lockit.Pitch);
                        texturescratch.UnlockRect((uint)level);
                    }
                    Direct3D9Helper.device.UpdateTexture(texturescratch, texture);
                    return texture;
                }
                else
                {
                    var asimage = DirectXHelper.GetImageConverter(material, 512, 512);
                    var texture = Direct3D9Helper.device.CreateTexture((uint)asimage.GetImageWidth(), (uint)asimage.GetImageHeight(), 1, 0U, D3DFormat.A8R8G8B8, D3DPool.Default, null);
                    var texturescratch = Direct3D9Helper.device.CreateTexture((uint)asimage.GetImageWidth(), (uint)asimage.GetImageHeight(), 1, 0U, D3DFormat.A8R8G8B8, D3DPool.SystemMemory, null);
                    D3DLockedRect lockit = texturescratch.LockRect(0);
                    DirectXHelper.ConvertToBitmap(asimage, lockit.Bits, asimage.GetImageWidth(), asimage.GetImageHeight(), lockit.Pitch);
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
                var data = DirectXHelper.ConvertToXYZNorDiffuseTex1(primitive);
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
        #endregion
    }
}