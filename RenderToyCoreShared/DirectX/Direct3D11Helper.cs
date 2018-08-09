////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToyCOM;
using RenderToy.Diagnostics;
using RenderToy.DocumentModel;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.Shaders;
using RenderToy.Textures;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace RenderToy.DirectX
{
    class Direct3D11Helper
    {
        public static ID3D11Device d3d11Device = Direct3D11.D3D11CreateDevice();
        public static ID3D11InputLayout d3d11InputLayout;
        public static ID3D11RasterizerState d3d11RasterizerState;
        public static ID3D11SamplerState d3d11SamplerState;
        public static Action<ID3D11DeviceContext4, Matrix3D, Matrix3D, string> CreateSceneDraw(SparseScene scene)
        {
            const int SIZEOF_CONSTANTBLOCK = 256;
            const int SIZEOF_MATRIX = 4 * 4 * 4;
            var d3d11constantbufferCPU = new byte[256 * 1024];
            ID3D11Buffer d3d11constantbufferGPU = null;
            unsafe
            {
                D3D11_SUBRESOURCE_DATA *subresource = null;
                Direct3D11Helper.d3d11Device.CreateBuffer(new D3D11_BUFFER_DESC { ByteWidth = (uint)d3d11constantbufferCPU.Length, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_CONSTANT_BUFFER, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = 4 * 16 }, ref *subresource, ref d3d11constantbufferGPU);
            }
            // We're collecting constant buffers because DX11 hates to do actual work.
            var constantbufferlist = new[] { d3d11constantbufferGPU };
            return (context, transformCamera, transformViewProjection, name) =>
            {
                string constantbufferblock = "Constant Buffer (" + name + ")";
                RenderToyEventSource.Default.MarkerBegin(constantbufferblock);
                int COUNT_OBJECTS = scene.IndexToNodePrimitive.Count;
                for (int i = 0; i < COUNT_OBJECTS; ++i)
                {
                    Matrix3D transformModel = scene.TableTransform[i];
                    var transformModelViewProjection = transformModel * transformViewProjection;
                    Buffer.BlockCopy(DirectXHelper.ConvertToD3DMatrix(transformModelViewProjection), 0, d3d11constantbufferCPU, i * SIZEOF_CONSTANTBLOCK, SIZEOF_MATRIX);
                    Buffer.BlockCopy(DirectXHelper.ConvertToD3DMatrix(transformCamera), 0, d3d11constantbufferCPU, i * SIZEOF_CONSTANTBLOCK + 1 * SIZEOF_MATRIX, SIZEOF_MATRIX);
                    Buffer.BlockCopy(DirectXHelper.ConvertToD3DMatrix(transformModel), 0, d3d11constantbufferCPU, i * SIZEOF_CONSTANTBLOCK + 2 * SIZEOF_MATRIX, SIZEOF_MATRIX);
                }
                RenderToyEventSource.Default.MarkerEnd(constantbufferblock);
                string commandbufferblock = "Command Buffer (" + name + ")";
                RenderToyEventSource.Default.MarkerBegin(commandbufferblock);
                var box = new D3D11_BOX { right = (uint)(SIZEOF_CONSTANTBLOCK * scene.TableTransform.Count), bottom = 1, back = 1};
                context.UpdateSubresource1(d3d11constantbufferGPU, 0, box, UnmanagedCopy.Create(d3d11constantbufferCPU), 0, 0, (uint)D3D11_COPY_FLAGS.D3D11_COPY_DISCARD);
                context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY.D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                context.IASetInputLayout(d3d11InputLayout);
                context.RSSetState(d3d11RasterizerState);
                context.PSSetSamplers(0, 1, d3d11SamplerState);
                for (int i = 0; i < COUNT_OBJECTS; ++i)
                {
                    var thistransformindex = scene.IndexToTransform[i];
                    var thisprimitive = scene.TableNodePrimitive[scene.IndexToNodePrimitive[i]];
                    var thismaterial = scene.TableNodeMaterial[scene.IndexToNodeMaterial[i]];
                    var vertexbuffer = Direct3D11Helper.CreateVertexBuffer(thisprimitive);
                    if (vertexbuffer == null) continue;
                    var objmat = thismaterial as LoaderOBJ.OBJMaterial;
                    var collecttextures = new[]
                    {
                        CreateTextureView(objmat == null ? thismaterial : objmat.map_Kd, StockMaterials.PlasticWhite),
                        CreateTextureView(objmat == null ? null : objmat.map_d, StockMaterials.PlasticWhite),
                        CreateTextureView(objmat == null ? null : objmat.map_bump, StockMaterials.PlasticLightBlue),
                        CreateTextureView(objmat == null ? null : objmat.displacement, StockMaterials.PlasticWhite)
                    };
                    {
                        var strides = (uint)(thistransformindex * SIZEOF_CONSTANTBLOCK / 16U);
                        var sizes = 5U * SIZEOF_MATRIX;
                        context.VSSetConstantBuffers1(0, 1, constantbufferlist[0], strides, sizes);
                    }
                    {
                        var strides = (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1));
                        var sizes = 0U;
                        context.IASetVertexBuffers(0, 1, vertexbuffer.d3d11Buffer, strides, sizes);
                    }
                    for (uint bufferIndex = 0; bufferIndex < collecttextures.Length; ++bufferIndex)
                    {
                        context.PSSetShaderResources(bufferIndex, 1, collecttextures[bufferIndex]);
                    }
                    context.Draw(vertexbuffer.vertexCount, 0);
                }
                RenderToyEventSource.Default.MarkerEnd(commandbufferblock);
            };
        }
        public static ID3D11ShaderResourceView CreateTextureView(IMaterial material, IMaterial missing)
        {
            if (material == null) material = missing;
            if (material == null) material = StockMaterials.Missing;
#if OPENVR_INSTALLED
            if (material is MaterialOpenVRCameraDistorted vr)
            {
                var dev = Marshal.GetComInterfaceForObject(d3d11Device, typeof(ID3D11Device));
                var srv = vr.TrackedCamera.GetVideoStreamTextureD3D11(vr.TrackedCameraHandle, VRTrackedCameraFrameType.Distorted, dev);
                return (ID3D11ShaderResourceView)Marshal.GetTypedObjectForIUnknown(srv, typeof(ID3D11Texture2D));
            }
#endif // OPENVR_INSTALLED
            return MementoServer.Default.Get(material, DX11TextureView, () =>
            {
                var astexture = material as ITexture;
                if (astexture != null)
                {
                    var pInitialData = new List<MIDL_D3D11_SUBRESOURCE_DATA>();
                    var retainMips = new List<UnmanagedCopy>();
                    for (int miplevel = 0; miplevel < astexture.GetTextureLevelCount(); ++miplevel)
                    {
                        var level = astexture.GetSurface(0, miplevel);
                        if (level == null) return null;
                        MIDL_D3D11_SUBRESOURCE_DATA FillpInitialData;
                        byte[] texturedata = new byte[4 * level.GetImageWidth() * level.GetImageHeight()];
                        DirectXHelper.ConvertToBitmap(level, Marshal.UnsafeAddrOfPinnedArrayElement(texturedata, 0), level.GetImageWidth(), level.GetImageHeight(), 4 * level.GetImageWidth());
                        var access = UnmanagedCopy.Create(texturedata);
                        retainMips.Add(access);
                        FillpInitialData.pSysMem = access;
                        FillpInitialData.SysMemPitch = (uint)(4 * level.GetImageWidth());
                        FillpInitialData.SysMemSlicePitch = (uint)(4 * level.GetImageWidth() * level.GetImageHeight());
                        pInitialData.Add(FillpInitialData);
                    }
                    var level0 = astexture.GetSurface(0, 0);
                    var desc = new D3D11_TEXTURE2D_DESC();
                    desc.Width = (uint)level0.GetImageWidth();
                    desc.Height = (uint)level0.GetImageHeight();
                    desc.MipLevels = (uint)pInitialData.Count;
                    desc.ArraySize = 1;
                    desc.Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM;
                    desc.SampleDesc.Count = 1;
                    desc.Usage = D3D11_USAGE.D3D11_USAGE_IMMUTABLE;
                    desc.BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_SHADER_RESOURCE;
                    ID3D11Texture2D texture = null;
                    var pInitialDataArray = pInitialData.ToArray();
                    D3D11Shim.Device_CreateTexture2D(d3d11Device, desc, pInitialDataArray, ref texture);
                    var vdesc = new D3D11_SHADER_RESOURCE_VIEW_DESC();
                    vdesc.Format = DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
                    vdesc.ViewDimension = D3D_SRV_DIMENSION.D3D11_SRV_DIMENSION_TEXTURE2D;
                    vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.Texture2D.MipLevels = (uint)pInitialData.Count;
                    vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.Texture2D.MostDetailedMip = 0;
                    ID3D11ShaderResourceView srview = null;
                    Direct3D11Helper.d3d11Device.CreateShaderResourceView(texture, vdesc, ref srview);
                    return srview;
                }
                else
                {
                    var asimage = DirectXHelper.GetImageConverter(material, 512, 512);
                    var desc = new D3D11_TEXTURE2D_DESC();
                    desc.Width = (uint)asimage.GetImageWidth();
                    desc.Height = (uint)asimage.GetImageHeight();
                    desc.MipLevels = 1;
                    desc.ArraySize = 1;
                    desc.Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM;
                    desc.SampleDesc.Count = 1;
                    desc.Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT;
                    desc.BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_SHADER_RESOURCE;
                    var pInitialData = new D3D11_SUBRESOURCE_DATA();
                    byte[] texturedata = new byte[4 * asimage.GetImageWidth() * asimage.GetImageHeight()];
                    DirectXHelper.ConvertToBitmap(asimage, Marshal.UnsafeAddrOfPinnedArrayElement(texturedata, 0), asimage.GetImageWidth(), asimage.GetImageHeight(), 4 * asimage.GetImageWidth());
                    pInitialData.pSysMem = UnmanagedCopy.Create(texturedata);
                    pInitialData.SysMemPitch = (uint)(4 * asimage.GetImageWidth());
                    pInitialData.SysMemSlicePitch = (uint)(4 * asimage.GetImageWidth() * asimage.GetImageHeight());
                    ID3D11Texture2D texture = null;
                    Direct3D11Helper.d3d11Device.CreateTexture2D(desc, pInitialData, ref texture);
                    var vdesc = new D3D11_SHADER_RESOURCE_VIEW_DESC();
                    vdesc.Format = DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
                    vdesc.ViewDimension = D3D_SRV_DIMENSION.D3D10_SRV_DIMENSION_TEXTURE2D;
                    vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.Texture2D.MipLevels = 1;
                    vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.Texture2D.MostDetailedMip = 0;
                    ID3D11ShaderResourceView srview = null;
                    Direct3D11Helper.d3d11Device.CreateShaderResourceView(texture, vdesc, ref srview);
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
                ID3D11Buffer d3d11Buffer = null;
                Direct3D11Helper.d3d11Device.CreateBuffer(
                    new D3D11_BUFFER_DESC { ByteWidth = size, Usage = D3D11_USAGE.D3D11_USAGE_IMMUTABLE, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_VERTEX_BUFFER, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = (uint)Marshal.SizeOf(typeof(XYZ)) },
                    new D3D11_SUBRESOURCE_DATA { pSysMem = UnmanagedCopy.Create(verticesout), SysMemPitch = 0, SysMemSlicePitch = 0 },
                    ref d3d11Buffer);
                return new VertexBufferInfo { d3d11Buffer = d3d11Buffer, vertexCount = (uint)verticesout.Length };
            });
        }
        public class VertexBufferInfo
        {
            public ID3D11Buffer d3d11Buffer;
            public uint vertexCount;
        }
        static Direct3D11Helper()
        {
            var inputelements = new[]
            {
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "POSITION", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Position").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "NORMAL", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Normal").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "COLOR", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Diffuse").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "TEXCOORD", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("TexCoord").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "TANGENT", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Tangent").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "BINORMAL", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Bitangent").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
            };
            D3D11Shim.Device_CreateInputLayout(d3d11Device, ref inputelements, UnmanagedCopy.Create(HLSL.D3D11VS), HLSL.D3D11VS.Length, ref d3d11InputLayout);
            d3d11Device.CreateRasterizerState(new D3D11_RASTERIZER_DESC { FillMode = D3D11_FILL_MODE.D3D11_FILL_SOLID, CullMode = D3D11_CULL_MODE.D3D11_CULL_NONE }, ref d3d11RasterizerState);
            {
                var pSamplerDesc = new D3D11_SAMPLER_DESC();
                pSamplerDesc.Filter = D3D11_FILTER.D3D11_FILTER_MIN_MAG_MIP_LINEAR;
                pSamplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_MODE.D3D11_TEXTURE_ADDRESS_WRAP;
                pSamplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_MODE.D3D11_TEXTURE_ADDRESS_WRAP;
                pSamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_MODE.D3D11_TEXTURE_ADDRESS_WRAP;
                pSamplerDesc.MipLODBias = 0;
                pSamplerDesc.MaxAnisotropy = 0;
                pSamplerDesc.ComparisonFunc = D3D11_COMPARISON_FUNC.D3D11_COMPARISON_ALWAYS;
                pSamplerDesc.BorderColor = new float[4];
                pSamplerDesc.BorderColor[0] = 0;
                pSamplerDesc.BorderColor[1] = 0;
                pSamplerDesc.BorderColor[2] = 0;
                pSamplerDesc.BorderColor[3] = 0;
                pSamplerDesc.MinLOD = 0;
                pSamplerDesc.MaxLOD = float.MaxValue;
                d3d11Device.CreateSamplerState(pSamplerDesc, ref d3d11SamplerState);
            }
        }
        readonly static string DX11TextureView = "DX11TextureView";
        readonly static string DX11VertexBuffer = "DX11VertexBuffer";
    }
}