////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToyCOM;
using RenderToy.Diagnostics;
using RenderToy.DocumentModel;
using RenderToy.Expressions;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.Shaders;
using RenderToy.Textures;
using RenderToy.Utility;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace RenderToy.DirectX
{
    public static class Direct3D11Helper
    {
        #region - Section : Public Fields & Methods -
        /// <summary>
        /// Initialize D3D11 and all common resources.
        /// IMPORTANT: Call this BEFORE using this class.
        /// DO NOT CALL THIS FROM A STATIC INITIALIZER.
        /// </summary>
        public static void Initialize()
        {
            Dispatcher = DispatcherHelper.CreateDispatcher("Direct3D11 Synchronized Dispatcher");
            Dispatcher.Invoke(() =>
            {
                d3d11Device = Direct3D11.D3D11CreateDevice();
            });
            var inputelements = new[]
            {
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "POSITION", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Position").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "NORMAL", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Normal").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "COLOR", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Diffuse").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "TEXCOORD", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("TexCoord").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "TANGENT", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Tangent").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new MIDL_D3D11_INPUT_ELEMENT_DESC { SemanticName = "BINORMAL", SemanticIndex = 0, Format = DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT, InputSlot = 0, AlignedByteOffset = (uint)Marshal.OffsetOf<XYZNorDiffuseTex1>("Bitangent").ToInt32(), InputSlotClass = D3D11_INPUT_CLASSIFICATION.D3D11_INPUT_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
            };
            Dispatcher.Invoke(() => { D3D11Shim.Device_CreateInputLayout(d3d11Device, ref inputelements, UnmanagedCopy.Create(HLSL.D3D11VS), HLSL.D3D11VS.Length, ref d3d11InputLayout); });
            Dispatcher.Invoke(() => { d3d11Device.CreateRasterizerState(new D3D11_RASTERIZER_DESC { FillMode = D3D11_FILL_MODE.D3D11_FILL_SOLID, CullMode = D3D11_CULL_MODE.D3D11_CULL_NONE }, ref d3d11RasterizerState); });
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
                Dispatcher.Invoke(() => { d3d11Device.CreateSamplerState(pSamplerDesc, ref d3d11SamplerState); });
            }
            Dispatcher.Invoke(() =>
            {
                d3d11Device.CreateVertexShader(UnmanagedCopy.Create(HLSL.D3D11VS), (ulong)HLSL.D3D11VS.Length, null, ref d3d11VertexShader);
                d3d11Device.CreatePixelShader(UnmanagedCopy.Create(HLSL.D3D11PS), (ulong)HLSL.D3D11PS.Length, null, ref d3d11PixelShader);
                d3d11Device.CreatePixelShader(UnmanagedCopy.Create(HLSL.D3D11PSEnvironment), (ulong)HLSL.D3D11PSEnvironment.Length, null, ref d3d11PixelShaderEnvironment);
                d3d11Device.CreatePixelShader(UnmanagedCopy.Create(HLSL.D3D11PSUnlit), (ulong)HLSL.D3D11PSUnlit.Length, null, ref d3d11PixelShaderUnlit);
            });
        }
        /// <summary>
        /// Create a draw command for a scene.
        /// This will create a closure containing all the rendering resources
        /// required to draw the supplied scene. This closure is intended to
        /// be called from within the render loop after obtaining a command
        /// context.
        /// </summary>
        /// <param name="scene">The scene to be rendered.</param>
        /// <returns>A continuation which draws the scene in the given context
        /// with supplied constants.</returns>
        public static Action<ID3D11DeviceContext4, Dictionary<string, object>> CreateSceneDraw(SparseScene scene)
        {
            const int SIZEOF_CONSTANTBLOCK = 256;
            const int SIZEOF_MATRIX = 4 * 4 * 4;
            var d3d11constantbufferCPU = new byte[256 * 1024];
            ID3D11Buffer d3d11constantbufferGPU = null;
            // We're collecting constant buffers because DX11 hates to do actual work.
            ID3D11Buffer[] constantbufferlist = null;
            Dispatcher.Invoke(() =>
            {
                unsafe
                {
                    D3D11_SUBRESOURCE_DATA* subresource = null;
                    Direct3D11Helper.d3d11Device.CreateBuffer(new D3D11_BUFFER_DESC { ByteWidth = (uint)d3d11constantbufferCPU.Length, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_CONSTANT_BUFFER, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = 4 * 16 }, ref *subresource, ref d3d11constantbufferGPU);
                }
                constantbufferlist = new[] { d3d11constantbufferGPU };
            });
            return async (context, constants) =>
            {
                var profilingName = (string)constants["profilingName"];
                var transformAspect = (Matrix3D)constants["transformAspect"];
                var transformCamera = (Matrix3D)constants["transformCamera"];
                var transformView = (Matrix3D)constants["transformView"];
                var transformProjection = (Matrix3D)constants["transformProjection"];
                var transformTail = transformView * transformProjection * transformAspect;
                string constantbufferblock = "Constant Buffer (" + profilingName + ")";
                RenderToyEventSource.Default.MarkerBegin(constantbufferblock);
                int COUNT_OBJECTS = scene.IndexToNodePrimitive.Count;
                for (int i = 0; i < COUNT_OBJECTS; ++i)
                {
                    Matrix3D transformModel = scene.TableTransform[i];
                    var transformModelViewProjection = transformModel * transformTail;
                    Buffer.BlockCopy(Direct3DHelper.ConvertToD3DMatrix(transformModelViewProjection), 0, d3d11constantbufferCPU, i * SIZEOF_CONSTANTBLOCK, SIZEOF_MATRIX);
                    Buffer.BlockCopy(Direct3DHelper.ConvertToD3DMatrix(transformCamera), 0, d3d11constantbufferCPU, i * SIZEOF_CONSTANTBLOCK + 1 * SIZEOF_MATRIX, SIZEOF_MATRIX);
                    Buffer.BlockCopy(Direct3DHelper.ConvertToD3DMatrix(transformModel), 0, d3d11constantbufferCPU, i * SIZEOF_CONSTANTBLOCK + 2 * SIZEOF_MATRIX, SIZEOF_MATRIX);
                }
                RenderToyEventSource.Default.MarkerEnd(constantbufferblock);
                string commandbufferblock = "Command Buffer (" + profilingName + ")";
                RenderToyEventSource.Default.MarkerBegin(commandbufferblock);
                var box = new D3D11_BOX { right = (uint)(SIZEOF_CONSTANTBLOCK * scene.TableTransform.Count), bottom = 1, back = 1 };
                Direct3D11Helper.Dispatcher.Invoke(() =>
                {
                    context.UpdateSubresource1(d3d11constantbufferGPU, 0, box, UnmanagedCopy.Create(d3d11constantbufferCPU), 0, 0, (uint)D3D11_COPY_FLAGS.D3D11_COPY_DISCARD);
                    context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY.D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    context.IASetInputLayout(d3d11InputLayout);
                    context.RSSetState(d3d11RasterizerState);
                    context.PSSetSamplers(0, 1, d3d11SamplerState);
                    for (int i = 0; i < COUNT_OBJECTS; ++i)
                    {
                        var thisprimitive = scene.TableNodePrimitive[scene.IndexToNodePrimitive[i]];
                        var vertexbuffer = CreateVertexBufferAsync(thisprimitive);
                        if (vertexbuffer == null) continue;
                        var thismaterial = scene.TableNodeMaterial[scene.IndexToNodeMaterial[i]];
                        {
                            context.VSSetShader(CreateVertexShaderAsync(thismaterial), null, 0);
                            context.PSSetShader(CreatePixelShaderAsync(thismaterial), null, 0);
                        }
                        ID3D11ShaderResourceView[] collecttextures = CreateShaderResourceViews(thismaterial);
                        if (collecttextures == null) continue;
                        {
                            var thistransformindex = scene.IndexToTransform[i];
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
                });
            };
        }
        /// <summary>
        /// Bound threaded dispatcher for the D3D11 device.
        /// This is the only dispatcher which is guaranteed to be safe for any
        /// D3D11 call at any time.
        /// </summary>
        public static Dispatcher Dispatcher;
        /// <summary>
        /// The D3D11 device.
        /// Calls to this object should be marshaled through the D3D11 dispatcher.
        /// (Unless you REALLY know what you're doing).
        /// </summary>
        public static ID3D11Device d3d11Device;
        #endregion
        #region - Section : Common Resources -
        /// <summary>
        /// Common vertex layout defining position, normal, texture coordinate
        /// and tangent basis.
        /// </summary>
        static ID3D11InputLayout d3d11InputLayout;
        /// <summary>
        /// Common rasterizer state defining solid triangles.
        /// </summary>
        static ID3D11RasterizerState d3d11RasterizerState;
        /// <summary>
        /// Common sampler state defining a linear min-mag-mip sampling.
        /// </summary>
        static ID3D11SamplerState d3d11SamplerState;
        static ID3D11VertexShader d3d11VertexShader;
        static ID3D11PixelShader d3d11PixelShader;
        static ID3D11PixelShader d3d11PixelShaderEnvironment;
        static ID3D11PixelShader d3d11PixelShaderUnlit;
        static ID3D11ShaderResourceView d3d11TextureEnvironment;
        #endregion
        #region - Section : Texture Factory -
        /// <summary>
        /// This method is asynchronous by design.
        /// Create a shader resource view from a material definition.
        /// This will attempt to create the specified material or supply the
        /// missing material until it becomes available (if ever).
        /// </summary>
        /// <param name="material">The texture material to recreate.</param>
        /// <param name="missing">An alternative material to emit if the defintion is not ready.</param>
        /// <returns>A shader resource view.</returns>
        static ID3D11ShaderResourceView CreateShaderResourceViewAsync(IMaterial material, IMaterial missing)
        {
            if (material == null) material = missing;
            if (material == null) material = StockMaterials.Missing;
            if (material is DXGIDesktopMaterial)
            {
                return DXGIHelper.d3d11srv_copieddesktop;
            }
#if OPENVR_INSTALLED
            if (material is MaterialOpenVRCameraDistorted vr)
            {
                var dev = Marshal.GetComInterfaceForObject(d3d11Device, typeof(ID3D11Device));
                var srv = IntPtr.Zero;
                var header = new Valve.VR.CameraVideoStreamFrameHeader_t();
                OpenVRHelper.TrackedCamera.GetVideoStreamTextureD3D11(OpenVRHelper.TrackedCameraHandle, Valve.VR.EVRTrackedCameraFrameType.Distorted, dev, ref srv, ref header, (uint)Marshal.SizeOf(header));
                if (srv == IntPtr.Zero)
                {
                    return CreateShaderResourceViewSyncCached(missing);
                }
                return (ID3D11ShaderResourceView)Marshal.GetTypedObjectForIUnknown(srv, typeof(ID3D11Texture2D));
            }
#endif // OPENVR_INSTALLED
            ID3D11ShaderResourceView find;
            if (!generatedTextures.TryGetValue(material, out find))
            {
                Task.Run(() =>
                {
                    var createit = CreateShaderResourceViewSyncCached(material);
                    generatedTextures.AddOrUpdate(material, createit, (m, srv) => createit);
                });
                find = CreateShaderResourceViewSyncCached(missing);
                generatedTextures.AddOrUpdate(material, find, (m, srv) => find);
            }
            return find;
        }
        /// <summary>
        /// Create a shader resource view from a specific material.
        /// This function can fail if the material cannot be converted.
        /// </summary>
        /// <param name="material">The texture material to recreate.</param>
        /// <returns>A shader resource view representing this texture.</returns>
        static ID3D11ShaderResourceView CreateShaderResourceViewSyncCached(IMaterial material)
        {
            return MementoServer.Default.Get(material, DX11TextureView, () =>
            {
                if (material is ITexture texture)
                {
                    return CreateShaderResourceViewSyncUncachedFromTexture(texture);
                }
                else
                {
                    return CreateShaderResourceViewSyncUncachedFromSurface(material.GetImageConverter(512, 512));
                }
            });
        }
        /// <summary>
        /// Create an uncached shader resource view from a texture.
        /// </summary>
        /// <param name="texture">The texture to construct the shader resource view from.</param>
        /// <returns>A shader resource view corresponding to this texture.</returns>
        static ID3D11ShaderResourceView CreateShaderResourceViewSyncUncachedFromTexture(ITexture texture)
        {
            var freePin = new List<GCHandle>();
            try
            {
                var level0 = texture.GetSurface(0, 0);
                if (level0 == null) return null;
                // TODO: This is a bad way to detect cubemaps.
                // There needs to be some way to indicate that a texture is
                // expected to become a cubemap in the engine.
                if (texture.GetTextureArrayCount() == 6)
                {
                    var format = level0.GetFormat();
                    var pixelsize = Direct3DHelper.GetPixelSize(format);
                    var pInitialData = new List<MIDL_D3D11_SUBRESOURCE_DATA>();
                    for (int face = 0; face < 6; ++face)
                    {
                        var level = texture.GetSurface(face, 0);
                        if (level == null) return null;
                        MIDL_D3D11_SUBRESOURCE_DATA FillpInitialData;
                        var copy = level.Copy();
                        var pin = GCHandle.Alloc(copy, GCHandleType.Pinned);
                        freePin.Add(pin);
                        FillpInitialData.pSysMem = Marshal.UnsafeAddrOfPinnedArrayElement(copy, 0);
                        FillpInitialData.SysMemPitch = (uint)(pixelsize * level.GetImageWidth());
                        FillpInitialData.SysMemSlicePitch = (uint)(pixelsize * level.GetImageWidth() * level.GetImageHeight());
                        pInitialData.Add(FillpInitialData);
                    }
                    var desc = new D3D11_TEXTURE2D_DESC();
                    desc.Width = (uint)level0.GetImageWidth();
                    desc.Height = (uint)level0.GetImageHeight();
                    desc.MipLevels = 1;
                    desc.ArraySize = 6;
                    desc.Format = format;
                    desc.SampleDesc.Count = 1;
                    desc.Usage = D3D11_USAGE.D3D11_USAGE_IMMUTABLE;
                    desc.BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_SHADER_RESOURCE;
                    desc.MiscFlags = (uint)D3D11_RESOURCE_MISC_FLAG.D3D11_RESOURCE_MISC_TEXTURECUBE;
                    ID3D11Texture2D d3d11texture = null;
                    var pInitialDataArray = pInitialData.ToArray();
                    var vdesc = new D3D11_SHADER_RESOURCE_VIEW_DESC();
                    vdesc.Format = DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
                    vdesc.ViewDimension = D3D_SRV_DIMENSION.D3D11_SRV_DIMENSION_TEXTURECUBE;
                    vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.TextureCube.MipLevels = 1;
                    vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.TextureCube.MostDetailedMip = 0;
                    ID3D11ShaderResourceView srview = null;
                    Dispatcher.Invoke(() =>
                    {
                        D3D11Shim.Device_CreateTexture2D(d3d11Device, desc, pInitialDataArray, ref d3d11texture);
                        Direct3D11Helper.d3d11Device.CreateShaderResourceView(d3d11texture, vdesc, ref srview);
                    });
                    d3d11TextureEnvironment = srview;
                    return srview;
                }
                else
                {
                    var format = level0.GetFormat();
                    var pixelsize = Direct3DHelper.GetPixelSize(format);
                    var pInitialData = new List<MIDL_D3D11_SUBRESOURCE_DATA>();
                    for (int miplevel = 0; miplevel < texture.GetTextureLevelCount(); ++miplevel)
                    {
                        var level = texture.GetSurface(0, miplevel);
                        if (level == null) return null;
                        MIDL_D3D11_SUBRESOURCE_DATA FillpInitialData;
                        var copy = level.Copy();
                        var pin = GCHandle.Alloc(copy, GCHandleType.Pinned);
                        freePin.Add(pin);
                        FillpInitialData.pSysMem = Marshal.UnsafeAddrOfPinnedArrayElement(copy, 0);
                        FillpInitialData.SysMemPitch = (uint)(pixelsize * level.GetImageWidth());
                        FillpInitialData.SysMemSlicePitch = (uint)(pixelsize * level.GetImageWidth() * level.GetImageHeight());
                        pInitialData.Add(FillpInitialData);
                    }
                    var desc = new D3D11_TEXTURE2D_DESC();
                    desc.Width = (uint)level0.GetImageWidth();
                    desc.Height = (uint)level0.GetImageHeight();
                    desc.MipLevels = (uint)pInitialData.Count;
                    desc.ArraySize = 1;
                    desc.Format = texture.GetSurface(0, 0).GetFormat();
                    desc.SampleDesc.Count = 1;
                    desc.Usage = D3D11_USAGE.D3D11_USAGE_IMMUTABLE;
                    desc.BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_SHADER_RESOURCE;
                    ID3D11Texture2D d3d11texture = null;
                    var pInitialDataArray = pInitialData.ToArray();
                    var vdesc = new D3D11_SHADER_RESOURCE_VIEW_DESC();
                    vdesc.Format = DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
                    vdesc.ViewDimension = D3D_SRV_DIMENSION.D3D11_SRV_DIMENSION_TEXTURE2D;
                    vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.Texture2D.MipLevels = (uint)pInitialData.Count;
                    vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.Texture2D.MostDetailedMip = 0;
                    ID3D11ShaderResourceView srview = null;
                    Dispatcher.Invoke(() =>
                    {
                        D3D11Shim.Device_CreateTexture2D(d3d11Device, desc, pInitialDataArray, ref d3d11texture);
                        Direct3D11Helper.d3d11Device.CreateShaderResourceView(d3d11texture, vdesc, ref srview);
                    });
                    return srview;
                }
            }
            finally
            {
                foreach (var pin in freePin)
                {
                    pin.Free();
                }
            }
        }
        /// <summary>
        /// Create an uncached shader resource view from a flat image surface.
        /// </summary>
        /// <param name="surface">The image to construct.</param>
        /// <returns>A shader resource view for the given image.</returns>
        static ID3D11ShaderResourceView CreateShaderResourceViewSyncUncachedFromSurface(ISurface surface)
        {
            GCHandle? pin = null;
            try
            {
                var format = surface.GetFormat();
                int width = surface.GetImageWidth();
                int height = surface.GetImageHeight();
                var pixelsize = Direct3DHelper.GetPixelSize(format);
                var desc = new D3D11_TEXTURE2D_DESC();
                desc.Width = (uint)width;
                desc.Height = (uint)height;
                desc.MipLevels = 1;
                desc.ArraySize = 1;
                desc.Format = format;
                desc.SampleDesc.Count = 1;
                desc.Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT;
                desc.BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_SHADER_RESOURCE;
                var pInitialData = new D3D11_SUBRESOURCE_DATA();
                var data = surface.Copy();
                pin = GCHandle.Alloc(data, GCHandleType.Pinned);
                var ptr = Marshal.UnsafeAddrOfPinnedArrayElement(data, 0);
                pInitialData.pSysMem = ptr;
                pInitialData.SysMemPitch = (uint)(pixelsize * surface.GetImageWidth());
                pInitialData.SysMemSlicePitch = (uint)(pixelsize * surface.GetImageWidth() * surface.GetImageHeight());
                var vdesc = new D3D11_SHADER_RESOURCE_VIEW_DESC();
                vdesc.Format = DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
                vdesc.ViewDimension = D3D_SRV_DIMENSION.D3D10_SRV_DIMENSION_TEXTURE2D;
                vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.Texture2D.MipLevels = 1;
                vdesc.__MIDL____MIDL_itf_RenderToy_0005_00640002.Texture2D.MostDetailedMip = 0;
                ID3D11ShaderResourceView srview = null;
                Dispatcher.Invoke(() =>
                {
                    ID3D11Texture2D d3d11texture = null;
                    Direct3D11Helper.d3d11Device.CreateTexture2D(desc, pInitialData, ref d3d11texture);
                    Direct3D11Helper.d3d11Device.CreateShaderResourceView(d3d11texture, vdesc, ref srview);
                });
                return srview;
            }
            finally
            {
                if (pin.HasValue) pin.Value.Free();
            }
        }
        /// <summary>
        /// A concurrent dictionary of all currently available shader resource views.
        /// Changing this structure will immediately affect the render.
        /// </summary>
        static ConcurrentDictionary<IMaterial, ID3D11ShaderResourceView> generatedTextures = new ConcurrentDictionary<IMaterial, ID3D11ShaderResourceView>();
        /// <summary>
        /// A key used to look up created shader resource views in the global store.
        /// </summary>
        readonly static string DX11TextureView = "DX11TextureView";
        #endregion
        #region - Section : Vertex Buffer Factory -
        /// <summary>
        /// This method is asynchronous by design.
        /// Create a vertex buffer for a primitive. This will create the
        /// common vertex format for the primitive which includes
        /// positions, normals, texture coordinates and tangent basis.
        /// </summary>
        /// <param name="primitive">The primitive to construct.</param>
        /// <returns>A vertex buffer along with description of contents.</returns>
        static VertexBufferInfo CreateVertexBufferAsync(IPrimitive primitive)
        {
            if (primitive == null) return null;
            VertexBufferInfo find;
            if (!generatedVertexBuffers.TryGetValue(primitive, out find))
            {
                Task.Run(() =>
                {
                    var createit = CreateVertexBufferSyncCached(primitive);
                    generatedVertexBuffers.AddOrUpdate(primitive, createit, (m, srv) => createit);
                });
                find = null;
                generatedVertexBuffers.AddOrUpdate(primitive, find, (m, srv) => find);
            }
            return find;
        }
        /// <summary>
        /// Create a vertex buffer for a primitive.
        /// </summary>
        /// <param name="primitive">The primitive to create the vertex buffer from.</param>
        /// <returns>A vertex buffer descriptor.</returns>
        static VertexBufferInfo CreateVertexBufferSyncCached(IPrimitive primitive)
        {
            if (primitive == null) return null;
            return MementoServer.Default.Get(primitive, DX11VertexBuffer, () =>
            {
                var verticesout = Direct3DHelper.ConvertToXYZNorDiffuseTex1(primitive);
                if (verticesout.Length == 0) return null;
                var size = (uint)(Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) * verticesout.Length);
                ID3D11Buffer d3d11Buffer = null;
                Dispatcher.Invoke(() =>
                {
                    Direct3D11Helper.d3d11Device.CreateBuffer(
                        new D3D11_BUFFER_DESC { ByteWidth = size, Usage = D3D11_USAGE.D3D11_USAGE_IMMUTABLE, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_VERTEX_BUFFER, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = (uint)Marshal.SizeOf(typeof(XYZ)) },
                        new D3D11_SUBRESOURCE_DATA { pSysMem = UnmanagedCopy.Create(verticesout), SysMemPitch = 0, SysMemSlicePitch = 0 },
                        ref d3d11Buffer);
                });
                return new VertexBufferInfo { d3d11Buffer = d3d11Buffer, vertexCount = (uint)verticesout.Length };
            });
        }
        class VertexBufferInfo
        {
            internal ID3D11Buffer d3d11Buffer;
            internal uint vertexCount;
        }
        readonly static ConcurrentDictionary<IPrimitive, VertexBufferInfo> generatedVertexBuffers = new ConcurrentDictionary<IPrimitive, VertexBufferInfo>();
        readonly static string DX11VertexBuffer = "DX11VertexBuffer";
        #endregion
        #region - Section : Shader Factory -
        /// <summary>
        /// This method is asynchronous by design.
        /// Create a vertex shader for a given material.
        /// </summary>
        /// <param name="material">The material to generate the vertex shader from.</param>
        /// <returns>A vertex shader object.</returns>
        static ID3D11VertexShader CreateVertexShaderAsync(IMaterial material)
        {
            if (material == null) return null;
            ID3D11VertexShader find = null;
            if (!generatedVertexShaders.TryGetValue(material, out find))
            {
                Task.Run(() =>
                {
                    var createit = CreateVertexShaderSyncUncached(material);
                    generatedVertexShaders.AddOrUpdate(material, createit, (m, srv) => createit);
                });
                find = d3d11VertexShader;
                generatedVertexShaders.AddOrUpdate(material, find, (m, srv) => find);
            }
            return find;
        }
        /// <summary>
        /// Create a vertex shader from a given material.
        /// </summary>
        /// <param name="material">The material to create the vertex shader from.</param>
        /// <returns>A vertex shader object.</returns>
        static ID3D11VertexShader CreateVertexShaderSyncUncached(IMaterial material)
        {
            if (material is OBJMaterial)
            {
                return d3d11VertexShader;
            }
            if (material is IMNNode node)
            {
                var hlsl = node.GenerateHLSL();
                var bytecode = HLSLExtensions.CompileHLSL(hlsl, "vs", "vs_5_0");
                ID3D11VertexShader shader = null;
                Dispatcher.Invoke(() =>
                {
                    d3d11Device.CreateVertexShader(UnmanagedCopy.Create(bytecode), (ulong)bytecode.Length, null, ref shader);
                });
                return shader;
            }
            // TODO: For now we're just going to return the default always.
            return d3d11VertexShader;
        }
        /// <summary>
        /// A concurrent dictionary of all currently available vertex shaders.
        /// Changing this structure will immediately affect the render.
        /// </summary>
        readonly static ConcurrentDictionary<IMaterial, ID3D11VertexShader> generatedVertexShaders = new ConcurrentDictionary<IMaterial, ID3D11VertexShader>();
        /// <summary>
        /// A key used to look up created vertex shaders in the global store.
        /// </summary>
        readonly static string D3D11VertexShader = "D3D11VertexShader";
        #endregion
        #region - Section : Pixel Shader Factory -
        /// <summary>
        /// This method is asynchronous by design.
        /// Create a pixel shader for a given material.
        /// </summary>
        /// <param name="material">The material to generate the pixel shader from.</param>
        /// <returns>A pixel shader object.</returns>
        static ID3D11PixelShader CreatePixelShaderAsync(IMaterial material)
        {
            if (material == null) return null;
            ID3D11PixelShader find = null;
            if (!generatedPixelShaders.TryGetValue(material, out find))
            {
                Task.Run(() =>
                {
                    var createit = CreatePixelShaderSyncUncached(material);
                    generatedPixelShaders.AddOrUpdate(material, createit, (m, srv) => createit);
                });
                find = d3d11PixelShader;
                generatedPixelShaders.AddOrUpdate(material, find, (m, srv) => find);
            }
            return find;
        }
        /// <summary>
        /// Create a pixel shader from a given material.
        /// </summary>
        /// <param name="material">The material to create the pixel shader from.</param>
        /// <returns>A pixel shader object.</returns>
        static ID3D11PixelShader CreatePixelShaderSyncUncached(IMaterial material)
        {
            if (material is OBJMaterial)
            {
                return d3d11PixelShader;
            }
            if (material is DXGIDesktopMaterial)
            {
                return d3d11PixelShaderUnlit;
            }
            // TODO: This is a bad way to detect cubemaps.
            // There needs to be some way to indicate that a texture is
            // expected to become a cubemap in the engine.
            if (material is ITexture texture && texture.GetTextureArrayCount() == 6)
            {
                return d3d11PixelShaderEnvironment;
            }
            if (material is IMNNode node)
            {
                var hlsl = node.GenerateHLSL();
                var bytecode = HLSLExtensions.CompileHLSL(hlsl, "ps", "ps_5_0");
                ID3D11PixelShader shader = null;
                Dispatcher.Invoke(() =>
                {
                    d3d11Device.CreatePixelShader(UnmanagedCopy.Create(bytecode), (ulong)bytecode.Length, null, ref shader);
                });
                return shader;
            }
            // TODO: For now we're just going to return the default always.
            return d3d11PixelShader;
        }
        /// <summary>
        /// A concurrent dictionary of all currently available pixel shaders.
        /// Changing this structure will immediately affect the render.
        /// </summary>
        readonly static ConcurrentDictionary<IMaterial, ID3D11PixelShader> generatedPixelShaders = new ConcurrentDictionary<IMaterial, ID3D11PixelShader>();
        /// <summary>
        /// A key used to look up created pixel shaders in the global store.
        /// </summary>
        readonly static string D3D11PixelShader = "D3D11PixelShader";
        #endregion
        #region - Section : Texture Sets -
        static ID3D11ShaderResourceView[] CreateShaderResourceViews(IMaterial thismaterial)
        {
            if (thismaterial is GenericMaterial)
            {
                return new[]
                {
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticLightBlue),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    d3d11TextureEnvironment,
                };
            }
            else if (thismaterial is OBJMaterial objmat)
            {
                return new[]
                {
                    CreateShaderResourceViewAsync(objmat.map_Kd, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(objmat.map_d, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(objmat.map_bump, StockMaterials.PlasticLightBlue),
                    CreateShaderResourceViewAsync(objmat.displacement, StockMaterials.PlasticWhite),
                    d3d11TextureEnvironment,
                };
            }
            else if (thismaterial is SurfaceCrossToCube)
            {
                return new[]
                {
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticLightBlue),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(thismaterial, null),
                };
            }
            else if (thismaterial is DXGIDesktopMaterial)
            {
                return new[]
                {
                    CreateShaderResourceViewAsync(thismaterial, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticLightBlue),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    d3d11TextureEnvironment,
                };
            }
            else if (thismaterial is ISurface)
            {
                return new[]
                {
                    CreateShaderResourceViewAsync(thismaterial, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticLightBlue),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    d3d11TextureEnvironment,
                };
            }
            else if (thismaterial is ITexture)
            {
                return new[]
                {
                    CreateShaderResourceViewAsync(thismaterial, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticLightBlue),
                    CreateShaderResourceViewAsync(null, StockMaterials.PlasticWhite),
                    d3d11TextureEnvironment,
                };
            }
            throw new NotSupportedException("Can't build a texture list for this material.");
        }
        #endregion
    }
}