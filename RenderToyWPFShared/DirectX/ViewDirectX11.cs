using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Shaders;
using RenderToy.Textures;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewDirectX11 : FrameworkElement
    {
        public static DependencyProperty VertexShaderProperty = DependencyProperty.Register("VertexShader", typeof(byte[]), typeof(ViewDirectX11), new FrameworkPropertyMetadata(HLSL.D3D11VS, OnVertexShaderChanged));
        public byte[] VertexShader
        {
            get { return (byte[])GetValue(VertexShaderProperty); }
            set { SetValue(VertexShaderProperty, value); }
        }
        private static void OnVertexShaderChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ((ViewDirectX11)sender).SetVertexShader((byte[])e.NewValue);
        }
        public static DependencyProperty PixelShaderProperty = DependencyProperty.Register("PixelShader", typeof(byte[]), typeof(ViewDirectX11), new FrameworkPropertyMetadata(HLSL.D3D11PS, OnPixelShaderChanged));
        public byte[] PixelShader
        {
            get { return (byte[])GetValue(PixelShaderProperty); }
            set { SetValue(PixelShaderProperty, value); }
        }
        private static void OnPixelShaderChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ((ViewDirectX11)sender).SetPixelShader((byte[])e.NewValue);
        }
        static ViewDirectX11()
        {
            AttachedView.SceneProperty.OverrideMetadata(typeof(ViewDirectX11), new FrameworkPropertyMetadata(null, (s, e) =>
            {
                ((ViewDirectX11)s).RenderDX();
            }));
            AttachedView.TransformModelViewProjectionProperty.OverrideMetadata(typeof(ViewDirectX11), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) =>
            {
                ((ViewDirectX11)s).RenderDX();
            }));
            d3d11InputLayout = d3d11Device.CreateInputLayout(new[]
            {
                new D3D11InputElementDesc { SemanticName = "POSITION", SemanticIndex = 0, Format = DXGIFormat.R32G32B32_Float, InputSlot = 0, AlignedByteOffset = 0, InputSlotClass = D3D11InputClassification.PerVertexData, InstanceDataStepRate = 0 },
                new D3D11InputElementDesc { SemanticName = "NORMAL", SemanticIndex = 0, Format = DXGIFormat.R32G32B32_Float, InputSlot = 0, AlignedByteOffset = 12, InputSlotClass = D3D11InputClassification.PerVertexData, InstanceDataStepRate = 0 },
                new D3D11InputElementDesc { SemanticName = "COLOR", SemanticIndex = 0, Format = DXGIFormat.B8G8R8A8_Unorm, InputSlot = 0, AlignedByteOffset = 24, InputSlotClass = D3D11InputClassification.PerVertexData, InstanceDataStepRate = 0 },
                new D3D11InputElementDesc { SemanticName = "TEXCOORD", SemanticIndex = 0, Format = DXGIFormat.R32G32_Float, InputSlot = 0, AlignedByteOffset = 28, InputSlotClass = D3D11InputClassification.PerVertexData, InstanceDataStepRate = 0 },
                new D3D11InputElementDesc { SemanticName = "TANGENT", SemanticIndex = 0, Format = DXGIFormat.R32G32B32_Float, InputSlot = 0, AlignedByteOffset = 36, InputSlotClass = D3D11InputClassification.PerVertexData, InstanceDataStepRate = 0 },
                new D3D11InputElementDesc { SemanticName = "BINORMAL", SemanticIndex = 0, Format = DXGIFormat.R32G32B32_Float, InputSlot = 0, AlignedByteOffset = 48, InputSlotClass = D3D11InputClassification.PerVertexData, InstanceDataStepRate = 0 },
            }, HLSL.D3D11VS);
            d3d11RasterizerState = d3d11Device.CreateRasterizerState(new D3D11RasterizerDesc { FillMode = D3D11FillMode.Solid, CullMode = D3D11CullMode.None });
            {
                D3D11SamplerDesc pSamplerDesc;
                pSamplerDesc.Filter = D3D11Filter.MinMagMipLinear;
                pSamplerDesc.AddressU = D3D11TextureAddressMode.Wrap;
                pSamplerDesc.AddressV = D3D11TextureAddressMode.Wrap;
                pSamplerDesc.AddressW = D3D11TextureAddressMode.Wrap;
                pSamplerDesc.MipLODBias = 0;
                pSamplerDesc.MaxAnisotropy = 0;
                pSamplerDesc.ComparisonFunc = D3D11ComparisonFunc.Always;
                pSamplerDesc.BorderColor0 = 0;
                pSamplerDesc.BorderColor1 = 0;
                pSamplerDesc.BorderColor2 = 0;
                pSamplerDesc.BorderColor3 = 0;
                pSamplerDesc.MinLOD = 0;
                pSamplerDesc.MaxLOD = 0;
                d3d11SamplerState = d3d11Device.CreateSamplerState(pSamplerDesc);
            }
        }
        public ViewDirectX11()
        {
            d3d11constantbufferCPU = new byte[256 * 1024];
            d3d11constantbufferGPU = d3d11Device.CreateBuffer(new D3D11BufferDesc { ByteWidth = (uint)d3d11constantbufferCPU.Length, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.ConstantBuffer, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = 4 * 16 }, null);
        }
        void RenderDX()
        {
            if (d3d11VertexShader == null || d3d11PixelShader == null || d3d11DepthStencilView == null || d3d11RenderTargetView == null) return;
            var contextold = d3d11Device.GetImmediateContext();
            var context = contextold.QueryInterfaceD3D11DeviceContext4();
            int width = d3d11Texture2D_RT.GetWidth();
            int height = d3d11Texture2D_RT.GetHeight();
            RenderToyETWEventSource.Default.BeginFrame();
            context.ClearDepthStencilView(d3d11DepthStencilView, D3D11ClearFlag.Depth, 1, 0);
            context.ClearRenderTargetView(d3d11RenderTargetView, 0, 0, 0, 0);
            context.IASetPrimitiveTopology(D3DPrimitiveTopology.TriangleList);
            context.IASetInputLayout(d3d11InputLayout);
            context.VSSetShader(d3d11VertexShader);
            context.PSSetShader(d3d11PixelShader);
            context.RSSetState(d3d11RasterizerState);
            context.RSSetScissorRects(new[] { new D3D11Rect { left = 0, top = 0, right = width, bottom = height } });
            context.RSSetViewports(new[] { new D3D11Viewport { TopLeftX = 0, TopLeftY = 0, Width = width, Height = height, MinDepth = 0, MaxDepth = 1 } });
            context.OMSetRenderTargets(new[] { d3d11RenderTargetView }, d3d11DepthStencilView);
            context.PSSetSamplers(0, new[] { d3d11SamplerState });
            ////////////////////////////////////////////////////////////////////////////////
            // Draw the scene.
            var transformViewProjection = AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            // Assemble the scene parts.
            var scene = TransformedObject.Enumerate(AttachedView.GetScene(this)).ToArray();
            // We're collecting constant buffers because DX11 hates to do actual work.
            int constantbufferoffset = 0;
            List<uint> offsets = new List<uint>();
            foreach (var transformedobject in scene)
            {
                var transformModel = transformedobject.Transform;
                var transformModelViewProjection = transformModel * transformViewProjection;
                var vertexbuffer = CreateVertexBuffer(transformedobject.Node.Primitive);
                if (vertexbuffer == null) continue;
                offsets.Add((uint)constantbufferoffset);
                Buffer.BlockCopy(DirectXHelper.ConvertToD3DMatrix(transformModelViewProjection), 0, d3d11constantbufferCPU, constantbufferoffset, 4 * 16);
                constantbufferoffset += 4 * 16;
                // Pad up to 256 bytes.
                if ((constantbufferoffset & 0xFF) != 0)
                {
                    constantbufferoffset = constantbufferoffset & (~0xFF);
                    constantbufferoffset += 256;
                }
            }
            context.UpdateSubresource1(d3d11constantbufferGPU, 0, new D3D11Box { right = (uint)constantbufferoffset }, d3d11constantbufferCPU, 0, 0, D3D11CopyFlags.Discard);
            ////////////////////////////////////////////////////////////////////////////////
            // Start drawing.
            int constantbufferindex = 0;
            var constantoffset = new uint[1];
            var constantsize = new uint[1];
            var constantbufferlist = new[] { d3d11constantbufferGPU };
            foreach (var transformedobject in scene)
            {
                var vertexbuffer = CreateVertexBuffer(transformedobject.Node.Primitive);
                if (vertexbuffer == null) continue;
                constantoffset[0] = offsets[constantbufferindex] / 16;
                constantsize[0] = 4 * 16;
                context.VSSetConstantBuffers1(0, constantbufferlist, constantoffset, constantsize);
                ++constantbufferindex;
                var objmat = transformedobject.Node.Material as LoaderOBJ.OBJMaterial;
                context.PSSetShaderResources(0, new[]
                {
                    CreateTextureView(objmat == null ? transformedobject.Node.Material : objmat.map_Kd, StockMaterials.PlasticWhite),
                    CreateTextureView(objmat == null ? null : objmat.map_d, StockMaterials.PlasticWhite),
                    CreateTextureView(objmat == null ? null : objmat.map_bump, StockMaterials.PlasticLightBlue),
                    CreateTextureView(objmat == null ? null : objmat.displacement, StockMaterials.PlasticWhite)
                });
                context.IASetVertexBuffers(0, new[] { vertexbuffer.d3d11Buffer }, new[] { (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) }, new[] { 0U });
                context.Draw(vertexbuffer.vertexCount, 0);
            }
            context.Flush();
            RenderToyETWEventSource.Default.EndFrame();
            d3dimage.Lock();
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, d3d11Texture2D_RT.GetWidth(), d3d11Texture2D_RT.GetHeight()));
            d3dimage.Unlock();
        }
        void SetVertexShader(byte[] bytecode)
        {
            d3d11VertexShader = d3d11Device.CreateVertexShader(bytecode);
        }
        void SetPixelShader(byte[] bytecode)
        {
            d3d11PixelShader = d3d11Device.CreatePixelShader(bytecode);
        }
        class VertexBufferInfo
        {
            public D3D11Buffer d3d11Buffer;
            public uint vertexCount;
        }
        readonly static string DX11VertexBuffer = "DX11VertexBuffer";
        VertexBufferInfo CreateVertexBuffer(IPrimitive primitive)
        {
            if (primitive == null) return null;
            return MementoServer.Default.Get(primitive, DX11VertexBuffer, () =>
            {
                var verticesout = DirectXHelper.ConvertToXYZNorDiffuseTex1(primitive);
                if (verticesout.Length == 0) return null;
                var size = (uint)(Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) * verticesout.Length);
                var d3d11Buffer = d3d11Device.CreateBuffer(
                    new D3D11BufferDesc { ByteWidth = size, Usage = D3D11Usage.Immutable, BindFlags = D3D11BindFlag.VertexBuffer, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = (uint)Marshal.SizeOf(typeof(XYZ)) },
                    new D3D11SubresourceData { pSysMem = verticesout, SysMemPitch = 0, SysMemSlicePitch = 0 });
                return new VertexBufferInfo { d3d11Buffer = d3d11Buffer, vertexCount = (uint)verticesout.Length };
            });
        }
        readonly static string DX11TextureView = "DX11TextureView";
        D3D11ShaderResourceView CreateTextureView(IMaterial material, IMaterial missing)
        {
            if (material == null) material = missing;
            if (material == null) material = StockMaterials.Missing;
            return MementoServer.Default.Get(material, DX11TextureView, () =>
            {
                var astexture = material as ITexture;
                if (astexture != null)
                {
                    var level0 = astexture.GetTextureLevel(0);
                    if (level0 == null) return null;
                    D3D11Texture2DDesc desc = new D3D11Texture2DDesc();
                    desc.Width = (uint)level0.GetImageWidth();
                    desc.Height = (uint)level0.GetImageHeight();
                    desc.MipLevels = 1;
                    desc.ArraySize = 1;
                    desc.Format = DXGIFormat.B8G8R8A8_Unorm;
                    desc.SampleDesc.Count = 1;
                    desc.Usage = D3D11Usage.Immutable;
                    desc.BindFlags = D3D11BindFlag.ShaderResource;
                    D3D11SubresourceData pInitialData;
                    byte[] texturedata = new byte[4 * level0.GetImageWidth() * level0.GetImageHeight()];
                    MaterialBitmapConverter.ConvertToBitmap(level0, Marshal.UnsafeAddrOfPinnedArrayElement(texturedata, 0), level0.GetImageWidth(), level0.GetImageHeight(), 4 * level0.GetImageWidth());
                    pInitialData.pSysMem = texturedata;
                    pInitialData.SysMemPitch = (uint)(4 * level0.GetImageWidth());
                    pInitialData.SysMemSlicePitch = (uint)(4 * level0.GetImageWidth() * level0.GetImageHeight());
                    var texture = d3d11Device.CreateTexture2D(desc, pInitialData);
                    D3D11ShaderResourceViewDesc vdesc;
                    vdesc.Format = DXGIFormat.Unknown;
                    vdesc.ViewDimension = D3DSrvDimension.Texture2D;
                    vdesc.Texture2D.MipLevels = 1;
                    vdesc.Texture2D.MostDetailedMip = 0;
                    var srview = d3d11Device.CreateShaderResourceView(texture, vdesc);
                    return srview;
                }
                else
                {
                    var asimage = MaterialBitmapConverter.GetImageConverter(material, 512, 512);
                    D3D11Texture2DDesc desc = new D3D11Texture2DDesc();
                    desc.Width = (uint)asimage.GetImageWidth();
                    desc.Height = (uint)asimage.GetImageHeight();
                    desc.MipLevels = 1;
                    desc.ArraySize = 1;
                    desc.Format = DXGIFormat.B8G8R8A8_Unorm;
                    desc.SampleDesc.Count = 1;
                    desc.Usage = D3D11Usage.Default;
                    desc.BindFlags = D3D11BindFlag.ShaderResource;
                    D3D11SubresourceData pInitialData;
                    byte[] texturedata = new byte[4 * asimage.GetImageWidth() * asimage.GetImageHeight()];
                    MaterialBitmapConverter.ConvertToBitmap(asimage, Marshal.UnsafeAddrOfPinnedArrayElement(texturedata, 0), asimage.GetImageWidth(), asimage.GetImageHeight(), 4 * asimage.GetImageWidth());
                    pInitialData.pSysMem = texturedata;
                    pInitialData.SysMemPitch = (uint)(4 * asimage.GetImageWidth());
                    pInitialData.SysMemSlicePitch = (uint)(4 * asimage.GetImageWidth() * asimage.GetImageHeight());
                    var texture = d3d11Device.CreateTexture2D(desc, pInitialData);
                    D3D11ShaderResourceViewDesc vdesc;
                    vdesc.Format = DXGIFormat.Unknown;
                    vdesc.ViewDimension = D3DSrvDimension.Texture2D;
                    vdesc.Texture2D.MipLevels = 1;
                    vdesc.Texture2D.MostDetailedMip = 0;
                    var srview = d3d11Device.CreateShaderResourceView(texture, vdesc);
                    return srview;
                }
            });
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            NullablePtr<IntPtr> handle = new NullablePtr<IntPtr>(IntPtr.Zero);
            d3d9backbuffer = d3d9device.CreateRenderTarget((uint)availableSize.Width, (uint)availableSize.Height, D3DFormat.A8R8G8B8, D3DMultisample.None, 1, 0, handle);
            d3d11Texture2D_RT = d3d11Device.OpenSharedTexture2D(handle.Value);
            d3d11RenderTargetView = d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });
            var d3d11Texture2DDesc_DS = new D3D11Texture2DDesc { Width = (uint)availableSize.Width, Height = (uint)availableSize.Height, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.D32_Float, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.DepthStencil, CPUAccessFlags = 0 };
            d3d11Texture2D_DS = d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS, null);
            d3d11DepthStencilView = d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS, new D3D11DepthStencilViewDesc { Format = DXGIFormat.D32_Float, ViewDimension = D3D11DsvDimension.Texture2D, Texture2D = new D3D11Tex2DDsv { MipSlice = 0 } });
            d3dimage.Lock();
            d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, d3d9backbuffer.ManagedPtr);
            d3dimage.Unlock();
            RenderDX();
            return base.MeasureOverride(availableSize);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(d3dimage, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        // Direct3D9 Handling for D3DImage
        static Direct3D9Ex d3d9 = new Direct3D9Ex();
        static Form d3d9window = new Form();
        static Direct3DDevice9Ex d3d9device = d3d9.CreateDevice(d3d9window.Handle);
        Direct3DSurface9 d3d9backbuffer;
        D3DImage d3dimage = new D3DImage();
        // Direct3D11 Handling
        static D3D11Device d3d11Device = Direct3D11.D3D11CreateDevice();
        static D3D11InputLayout d3d11InputLayout;
        static D3D11RasterizerState d3d11RasterizerState;
        static D3D11SamplerState d3d11SamplerState;
        byte[] d3d11constantbufferCPU;
        D3D11Buffer d3d11constantbufferGPU;
        D3D11VertexShader d3d11VertexShader;
        D3D11PixelShader d3d11PixelShader;
        D3D11Texture2D d3d11Texture2D_RT;
        D3D11Texture2D d3d11Texture2D_DS;
        D3D11RenderTargetView d3d11RenderTargetView;
        D3D11DepthStencilView d3d11DepthStencilView;
    }
}