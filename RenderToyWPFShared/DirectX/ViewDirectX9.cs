////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Materials;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Textures;
using RenderToy.Utility;
using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public abstract class ViewDirectX9Base : FrameworkElement
    {
        #region - Section : Direct3D Resource Factory -
        protected ViewDirectX9Base()
        {
            d3dimage = new D3DImage();
            d3d = new Direct3D9();
            d3dimage.IsFrontBufferAvailableChanged += (s, e) =>
            {
                RecreateDevice();
            };
            RecreateDevice();
        }
        static readonly string GeneratedTextureToken = "DirectX9Texture";
        protected Direct3DTexture9 CreateTexture(IMaterial material, IMaterial missing)
        {
            if (material == null) material = missing;
            if (material == null) material = StockMaterials.Missing;
            return deviceboundmemento.Get(material, GeneratedTextureToken, () =>
            {
                var astexture = material as ITexture;
                if (astexture != null)
                {
                    var level0 = astexture.GetTextureLevel(0);
                    if (level0 == null) return null;
                    var texture = device.CreateTexture((uint)level0.GetImageWidth(), (uint)level0.GetImageHeight(), (uint)astexture.GetTextureLevelCount(), 0U, D3DFormat.A8R8G8B8, D3DPool.Managed);
                    for (int level = 0; level < astexture.GetTextureLevelCount(); ++level)
                    {
                        D3DLockedRect lockit = texture.LockRect((uint)level);
                        var thislevel = astexture.GetTextureLevel(level);
                        MaterialBitmapConverter.ConvertToBitmap(thislevel, lockit.Bits, thislevel.GetImageWidth(), thislevel.GetImageHeight(), lockit.Pitch);
                        texture.UnlockRect((uint)level);
                    }
                    return texture;
                }
                else
                {
                    var asimage = MaterialBitmapConverter.GetImageConverter(material, 512, 512);
                    var texture = device.CreateTexture((uint)asimage.GetImageWidth(), (uint)asimage.GetImageHeight(), 1, 0U, D3DFormat.A8R8G8B8, D3DPool.Managed);
                    D3DLockedRect lockit = texture.LockRect(0);
                    MaterialBitmapConverter.ConvertToBitmap(asimage, lockit.Bits, asimage.GetImageWidth(), asimage.GetImageHeight(), lockit.Pitch);
                    texture.UnlockRect(0);
                    return texture;
                }
            });
        }
        protected struct VertexBufferInfo
        {
            public Direct3DVertexBuffer9 VertexBuffer;
            public int PrimitiveCount;
        }
        static readonly string GeneratedVertexBufferToken = "DirectX9VertexBuffer";
        protected VertexBufferInfo CreateVertexBuffer(IPrimitive primitive)
        {
            if (primitive == null) return new VertexBufferInfo { PrimitiveCount = 0, VertexBuffer = null };
            return deviceboundmemento.Get(primitive, GeneratedVertexBufferToken, () =>
            {
                var data = DirectXHelper.ConvertToXYZNorDiffuseTex1(primitive);
                var size = (uint)(Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) * data.Length);
                VertexBufferInfo buffer = new VertexBufferInfo();
                if (data.Length > 0)
                {
                    buffer.VertexBuffer = device.CreateVertexBuffer(size, 0, 0U, D3DPool.Managed);
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
        void RecreateDevice()
        {
            deviceboundmemento.EvictByToken(GeneratedTextureToken);
            deviceboundmemento.EvictByToken(GeneratedVertexBufferToken);
            device = d3d.CreateDevice();
            RecreateSurfaces();
        }
        void RecreateSurfaces()
        {
            render_width = (int)ActualWidth;
            render_height = (int)ActualHeight;
            if (render_width == 0 || render_height == 0) return;
            rendertarget = device.CreateRenderTarget((uint)render_width, (uint)render_height, D3DFormat.A8R8G8B8, D3DMultisample.None, 0, 1);
            depthstencil = device.CreateDepthStencilSurface((uint)render_width, (uint)render_height, D3DFormat.D24X8, D3DMultisample.None, 0, 1);
            InvalidateVisual();
        }
        #endregion
        #region - Section : Overrides -
        protected abstract void RenderD3D();
        protected override void OnRender(DrawingContext drawingContext)
        {
            if (rendertarget == null || depthstencil == null) return;
            if (d3dimage.TryLock(new Duration(TimeSpan.FromMilliseconds(500))))
            {
                device.SetRenderTarget(0, rendertarget);
                device.SetDepthStencilSurface(depthstencil);
                device.BeginScene();
                device.Clear(D3DClear.Target | D3DClear.ZBuffer, 0x00000000, 1.0f, 0);
                device.SetRenderState(D3DRenderState.ZEnable, 1U);
                device.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
                device.SetRenderState(D3DRenderState.Lighting, 0);
                for (int i = 0; i < 8; ++i)
                {
                    device.SetSamplerState((uint)i, D3DSamplerState.MagFilter, (uint)D3DTextureFilter.Anisotropic);
                    device.SetSamplerState((uint)i, D3DSamplerState.MinFilter, (uint)D3DTextureFilter.Anisotropic);
                    device.SetSamplerState((uint)i, D3DSamplerState.MipFilter, (uint)D3DTextureFilter.Linear);
                    device.SetSamplerState((uint)i, D3DSamplerState.MaxAnisotropy, (uint)16);
                }
                var vertexdeclaration = device.CreateVertexDeclaration(new D3DVertexElement9[] {
                    new D3DVertexElement9 { Stream = 0, Offset = 0, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Position, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 12, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Normal, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 24, Type = D3DDeclType.D3DColor, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Color, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 28, Type = D3DDeclType.Float2, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.TexCoord, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 36, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Tangent, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 48, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Binormal, UsageIndex = 0 },
                });
                device.SetVertexDeclaration(vertexdeclaration);
                RenderD3D();
                device.EndScene();
                d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, rendertarget.ManagedPtr);
                d3dimage.AddDirtyRect(new Int32Rect(0, 0, render_width, render_height));
            }
            d3dimage.Unlock();
            drawingContext.DrawImage(d3dimage, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
        {
            RecreateSurfaces();
        }
        #endregion
        #region - Section : Private Fields -
        MementoServer deviceboundmemento = new MementoServer();
        D3DImage d3dimage;
        Direct3D9 d3d;
        protected Direct3DDevice9 device;
        Direct3DSurface9 rendertarget;
        Direct3DSurface9 depthstencil;
        int render_width;
        int render_height;
        #endregion
    }
    public class ViewDirectX9FixedFunction : ViewDirectX9Base
    {
        protected override void RenderD3D()
        {
            var mvp = AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            foreach (var transformedobject in TransformedObject.Enumerate(AttachedView.GetScene(this)))
            {
                var createdvertexbuffer = CreateVertexBuffer(transformedobject.Node.Primitive);
                if (createdvertexbuffer.VertexBuffer == null) continue;
                device.SetStreamSource(0, createdvertexbuffer.VertexBuffer, 0U, (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)));
                device.SetTexture(0, CreateTexture(transformedobject.Node.Material, null));
                device.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformedobject.Transform * mvp), 0));
                device.DrawPrimitive(D3DPrimitiveType.TriangleList, 0U, (uint)createdvertexbuffer.PrimitiveCount);
            }
        }
    }
    public class ViewDirectX9 : ViewDirectX9Base
    {
        public static DependencyProperty VertexShaderProperty = DependencyProperty.Register("VertexShader", typeof(byte[]), typeof(ViewDirectX9), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));
        public byte[] VertexShader
        {
            get { return (byte[])GetValue(VertexShaderProperty); }
            set { SetValue(VertexShaderProperty, value); }
        }
        public static DependencyProperty PixelShaderProperty = DependencyProperty.Register("PixelShader", typeof(byte[]), typeof(ViewDirectX9), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));
        public byte[] PixelShader
        {
            get { return (byte[])GetValue(PixelShaderProperty); }
            set { SetValue(PixelShaderProperty, value); }
        }
        protected override void RenderD3D()
        {
            if (VertexShader == null || PixelShader == null) return;
            var vertexshader = device.CreateVertexShader(VertexShader);
            var pixelshader = device.CreatePixelShader(PixelShader);
            device.SetVertexShader(vertexshader);
            device.SetPixelShader(pixelshader);
            var transformCamera = AttachedView.GetTransformCamera(this);
            var transformView = AttachedView.GetTransformView(this);
            var transformProjection = AttachedView.GetTransformProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            var transformViewProjection = transformView * transformProjection;
            foreach (var transformedobject in TransformedObject.Enumerate(AttachedView.GetScene(this)))
            {
                if (transformedobject?.Node?.Primitive == null) continue;
                var transformModel = transformedobject.Transform;
                var transformModelViewProjection = transformModel * transformViewProjection;
                var createdvertexbuffer = CreateVertexBuffer(transformedobject.Node.Primitive);
                if (createdvertexbuffer.VertexBuffer == null) continue;
                device.SetStreamSource(0, createdvertexbuffer.VertexBuffer, 0U, (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)));
                var objmat = transformedobject.Node.Material as LoaderOBJ.OBJMaterial;
                device.SetTexture(0, CreateTexture(objmat == null ? transformedobject.Node.Material : objmat.map_Kd, StockMaterials.PlasticWhite));
                device.SetTexture(1, CreateTexture(objmat == null ? null : objmat.map_d, StockMaterials.PlasticWhite));
                device.SetTexture(2, CreateTexture(objmat == null ? null : objmat.map_bump, StockMaterials.PlasticLightBlue));
                device.SetTexture(3, CreateTexture(objmat == null ? null : objmat.displacement, StockMaterials.PlasticWhite));
                device.SetVertexShaderConstantF(0, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformCamera), 0), 4);
                device.SetVertexShaderConstantF(4, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformModel), 0), 4);
                device.SetVertexShaderConstantF(8, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformView), 0), 4);
                device.SetVertexShaderConstantF(12, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformProjection), 0), 4);
                device.SetVertexShaderConstantF(16, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformModelViewProjection), 0), 4);
                device.DrawPrimitive(D3DPrimitiveType.TriangleList, 0U, (uint)createdvertexbuffer.PrimitiveCount);
            }
        }
    }
}