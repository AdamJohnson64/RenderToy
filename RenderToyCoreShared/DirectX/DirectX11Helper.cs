using RenderToy.DocumentModel;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Shaders;
using RenderToy.Textures;
using RenderToy.Transforms;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace RenderToy.DirectX
{
    class DirectX11Helper
    {
        public static D3D11Device d3d11Device = Direct3D11.D3D11CreateDevice();
        public static D3D11InputLayout d3d11InputLayout;
        public static D3D11RasterizerState d3d11RasterizerState;
        public static D3D11SamplerState d3d11SamplerState;
        public static Action<D3D11DeviceContext4, Matrix3D> CreateSceneDraw(SparseScene scene)
        {
            var d3d11constantbufferCPU = new byte[256 * 1024];
            var d3d11constantbufferGPU = DirectX11Helper.d3d11Device.CreateBuffer(new D3D11BufferDesc { ByteWidth = (uint)d3d11constantbufferCPU.Length, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.ConstantBuffer, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = 4 * 16 }, null);
            var execute_retransform = new List<Action<Matrix3D>>();
            var execute_drawprimitive = new List<Action<D3D11DeviceContext4>>();
            // We're collecting constant buffers because DX11 hates to do actual work.
            int constantbufferoffset = 0;
            {
                const int SIZEOF_MATRIX = 4 * 4 * 4;
                var constantbufferlist = new[] { d3d11constantbufferGPU };
                foreach (var transformedobject in scene)
                {
                    var vertexbuffer = DirectX11Helper.CreateVertexBuffer(transformedobject.NodePrimitive);
                    if (vertexbuffer == null) continue;
                    var thisconstantbufferoffset = constantbufferoffset;
                    execute_retransform.Add((transformViewProjection) =>
                    {
                        Matrix3D transformModel = transformedobject.TransformParent * transformedobject.NodeTransform.Transform;
                        var transformModelViewProjection = transformModel * transformViewProjection;
                        Buffer.BlockCopy(DirectXHelper.ConvertToD3DMatrix(transformModelViewProjection), 0, d3d11constantbufferCPU, thisconstantbufferoffset, SIZEOF_MATRIX);
                        Buffer.BlockCopy(DirectXHelper.ConvertToD3DMatrix(transformModel), 0, d3d11constantbufferCPU, thisconstantbufferoffset + 2 * SIZEOF_MATRIX, SIZEOF_MATRIX);
                    });
                    var objmat = transformedobject.NodeMaterial as LoaderOBJ.OBJMaterial;
                    execute_drawprimitive.Add((context2) =>
                    {
                        var collecttextures = new[]
                        {
                            CreateTextureView(objmat == null ? transformedobject.NodeMaterial : objmat.map_Kd, StockMaterials.PlasticWhite),
                            CreateTextureView(objmat == null ? null : objmat.map_d, StockMaterials.PlasticWhite),
                            CreateTextureView(objmat == null ? null : objmat.map_bump, StockMaterials.PlasticLightBlue),
                            CreateTextureView(objmat == null ? null : objmat.displacement, StockMaterials.PlasticWhite)
                        };
                        context2.VSSetConstantBuffers1(0, constantbufferlist, new[] { (uint)thisconstantbufferoffset / 16U }, new[] { 5U * SIZEOF_MATRIX });
                        context2.IASetVertexBuffers(0, new[] { vertexbuffer.d3d11Buffer }, new[] { (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) }, new[] { 0U });
                        context2.PSSetShaderResources(0, collecttextures);
                        context2.Draw(vertexbuffer.vertexCount, 0);
                    });
                    // Pad up to 256 bytes.
                    constantbufferoffset += 5 * SIZEOF_MATRIX;
                    if ((constantbufferoffset & 0xFF) != 0)
                    {
                        constantbufferoffset = constantbufferoffset & (~0xFF);
                        constantbufferoffset += 256;
                    }
                }
            }
            return (context, transformViewProjection) =>
            {
                foreach (var retransform in execute_retransform)
                {
                    retransform(transformViewProjection);
                }
                context.UpdateSubresource1(d3d11constantbufferGPU, 0, new D3D11Box { right = (uint)constantbufferoffset }, d3d11constantbufferCPU, 0, 0, D3D11CopyFlags.Discard);
                context.IASetPrimitiveTopology(D3DPrimitiveTopology.TriangleList);
                context.IASetInputLayout(d3d11InputLayout);
                context.RSSetState(d3d11RasterizerState);
                context.PSSetSamplers(0, new[] { d3d11SamplerState });
                foreach (var draw in execute_drawprimitive)
                {
                    draw(context);
                }
            };
        }
        public static D3D11ShaderResourceView CreateTextureView(IMaterial material, IMaterial missing)
        {
            if (material == null) material = missing;
            if (material == null) material = StockMaterials.Missing;
            if (material is MaterialOpenVRCameraDistorted vr)
            {
                return D3D11ShaderResourceView.WrapUnowned(vr.VRHost.TrackedCamera.GetVideoStreamTextureD3D11(vr.VRHost.TrackedCameraHandle, VRTrackedCameraFrameType.Distorted, d3d11Device.ManagedPtr));
            }
            return MementoServer.Default.Get(material, DX11TextureView, () =>
            {
                var astexture = material as ITexture;
                if (astexture != null)
                {
                    List<D3D11SubresourceData> pInitialData = new List<D3D11SubresourceData>();
                    for (int miplevel = 0; miplevel < astexture.GetTextureLevelCount(); ++miplevel)
                    {
                        var level = astexture.GetTextureLevel(miplevel);
                        if (level == null) return null;
                        D3D11SubresourceData FillpInitialData;
                        byte[] texturedata = new byte[4 * level.GetImageWidth() * level.GetImageHeight()];
                        DirectXHelper.ConvertToBitmap(level, Marshal.UnsafeAddrOfPinnedArrayElement(texturedata, 0), level.GetImageWidth(), level.GetImageHeight(), 4 * level.GetImageWidth());
                        FillpInitialData.pSysMem = texturedata;
                        FillpInitialData.SysMemPitch = (uint)(4 * level.GetImageWidth());
                        FillpInitialData.SysMemSlicePitch = (uint)(4 * level.GetImageWidth() * level.GetImageHeight());
                        pInitialData.Add(FillpInitialData);
                    }
                    var level0 = astexture.GetTextureLevel(0);
                    D3D11Texture2DDesc desc = new D3D11Texture2DDesc();
                    desc.Width = (uint)level0.GetImageWidth();
                    desc.Height = (uint)level0.GetImageHeight();
                    desc.MipLevels = (uint)pInitialData.Count;
                    desc.ArraySize = 1;
                    desc.Format = DXGIFormat.B8G8R8A8_Unorm;
                    desc.SampleDesc.Count = 1;
                    desc.Usage = D3D11Usage.Immutable;
                    desc.BindFlags = D3D11BindFlag.ShaderResource;
                    var texture = DirectX11Helper.d3d11Device.CreateTexture2D(desc, pInitialData.ToArray());
                    D3D11ShaderResourceViewDesc vdesc;
                    vdesc.Format = DXGIFormat.Unknown;
                    vdesc.ViewDimension = D3DSrvDimension.Texture2D;
                    vdesc.Texture2D.MipLevels = (uint)pInitialData.Count;
                    vdesc.Texture2D.MostDetailedMip = 0;
                    var srview = DirectX11Helper.d3d11Device.CreateShaderResourceView(texture, vdesc);
                    return srview;
                }
                else
                {
                    var asimage = DirectXHelper.GetImageConverter(material, 512, 512);
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
                    DirectXHelper.ConvertToBitmap(asimage, Marshal.UnsafeAddrOfPinnedArrayElement(texturedata, 0), asimage.GetImageWidth(), asimage.GetImageHeight(), 4 * asimage.GetImageWidth());
                    pInitialData.pSysMem = texturedata;
                    pInitialData.SysMemPitch = (uint)(4 * asimage.GetImageWidth());
                    pInitialData.SysMemSlicePitch = (uint)(4 * asimage.GetImageWidth() * asimage.GetImageHeight());
                    var texture = DirectX11Helper.d3d11Device.CreateTexture2D(desc, new[] { pInitialData });
                    D3D11ShaderResourceViewDesc vdesc;
                    vdesc.Format = DXGIFormat.Unknown;
                    vdesc.ViewDimension = D3DSrvDimension.Texture2D;
                    vdesc.Texture2D.MipLevels = 1;
                    vdesc.Texture2D.MostDetailedMip = 0;
                    var srview = DirectX11Helper.d3d11Device.CreateShaderResourceView(texture, vdesc);
                    return srview;
                }
            });
        }
        public static VertexBufferInfo CreateVertexBuffer(IPrimitive primitive)
        {
            if (primitive == null) return null;
            return MementoServer.Default.Get(primitive, DX11VertexBuffer, () =>
            {
                var verticesout = DirectXHelper.ConvertToXYZNorDiffuseTex1(primitive);
                if (verticesout.Length == 0) return null;
                var size = (uint)(Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) * verticesout.Length);
                var d3d11Buffer = DirectX11Helper.d3d11Device.CreateBuffer(
                    new D3D11BufferDesc { ByteWidth = size, Usage = D3D11Usage.Immutable, BindFlags = D3D11BindFlag.VertexBuffer, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = (uint)Marshal.SizeOf(typeof(XYZ)) },
                    new D3D11SubresourceData { pSysMem = verticesout, SysMemPitch = 0, SysMemSlicePitch = 0 });
                return new VertexBufferInfo { d3d11Buffer = d3d11Buffer, vertexCount = (uint)verticesout.Length };
            });
        }
        public class VertexBufferInfo
        {
            public D3D11Buffer d3d11Buffer;
            public uint vertexCount;
        }
        static DirectX11Helper()
        {
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
                pSamplerDesc.MaxLOD = float.MaxValue;
                d3d11SamplerState = d3d11Device.CreateSamplerState(pSamplerDesc);
            }
        }
        readonly static string DX11TextureView = "DX11TextureView";
        readonly static string DX11VertexBuffer = "DX11VertexBuffer";
    }
}