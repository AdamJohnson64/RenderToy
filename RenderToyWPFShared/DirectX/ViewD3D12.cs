using RenderToyCOM;
using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Expressions;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.Shaders;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewD3D12 : FrameworkElement
    {
        public static D3D12Device d3d12Device;
        static ViewD3D12()
        {
            AttachedView.SceneProperty.OverrideMetadata(typeof(ViewD3D12), new FrameworkPropertyMetadata(null, (s, e) =>
            {
                ((ViewD3D12)s).RenderDX();
            }));
            AttachedView.TransformModelViewProjectionProperty.OverrideMetadata(typeof(ViewD3D12), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) =>
            {
                ((ViewD3D12)s).RenderDX();
            }));
            Direct3D12.D3D12GetDebugInterface().EnableDebugLayer();
            d3d12Device = Direct3D12.D3D12CreateDevice();
        }
        public ViewD3D12()
        {
            d3d12CommandQueue = d3d12Device.CreateCommandQueue(new D3D12_COMMAND_QUEUE_DESC { Type = D3D12_COMMAND_LIST_TYPE.D3D12_COMMAND_LIST_TYPE_DIRECT });
            d3d12CommandAllocator = d3d12Device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE.D3D12_COMMAND_LIST_TYPE_DIRECT);
            ////////////////////////////////////////////////////////////////////////////////
            // Create the Vertex Shader and Pixel Shader
            var bytecode_VertexShader = HLSLExtensions.CompileHLSL(HLSL.D3D12Simple, "vs", "vs_5_0");
            var bytecode_PixelShader = HLSLExtensions.CompileHLSL(HLSL.D3D12Simple, "ps", "ps_5_0");
            ////////////////////////////////////////////////////////////////////////////////
            // Extract the Root Signature
            byte[] bytecode_RootSignature = ExtractRootSignature(bytecode_VertexShader);
            d3d12RootSignature = d3d12Device.CreateRootSignature(0, bytecode_RootSignature);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Pipeline State.
            var d3d12GraphicsPipelineStateDesc = new D3D12GraphicsPipelineStateDesc();
            d3d12GraphicsPipelineStateDesc.pRootSignature = d3d12RootSignature;
            d3d12GraphicsPipelineStateDesc.VS = bytecode_VertexShader;
            d3d12GraphicsPipelineStateDesc.PS = bytecode_PixelShader;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget = new D3D12_RENDER_TARGET_BLEND_DESC[8];
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].BlendEnable = 0;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND.D3D12_BLEND_ONE;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND.D3D12_BLEND_ZERO;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP.D3D12_BLEND_OP_ADD;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND.D3D12_BLEND_ONE;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND.D3D12_BLEND_ZERO;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP.D3D12_BLEND_OP_ADD;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].LogicOp = D3D12_LOGIC_OP.D3D12_LOGIC_OP_NOOP;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = 15;
            d3d12GraphicsPipelineStateDesc.SampleMask = uint.MaxValue;
            d3d12GraphicsPipelineStateDesc.RasterizerState.FillMode = D3D12_FILL_MODE.D3D12_FILL_MODE_SOLID;
            d3d12GraphicsPipelineStateDesc.RasterizerState.CullMode = D3D12_CULL_MODE.D3D12_CULL_MODE_NONE;
            d3d12GraphicsPipelineStateDesc.InputLayout.pInputElementDescs = new D3D12_INPUT_ELEMENT_DESC[]
            {
                new D3D12_INPUT_ELEMENT_DESC { SemanticName = "POSITION", SemanticIndex = 0, Format = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_FLOAT, InputSlot = 0, AlignedByteOffset = 0, InputSlotClass = D3D12_INPUT_CLASSIFICATION.D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new D3D12_INPUT_ELEMENT_DESC { SemanticName = "NORMAL", SemanticIndex = 0, Format = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_FLOAT, InputSlot = 0, AlignedByteOffset = 12, InputSlotClass = D3D12_INPUT_CLASSIFICATION.D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new D3D12_INPUT_ELEMENT_DESC { SemanticName = "COLOR", SemanticIndex = 0, Format = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, InputSlot = 0, AlignedByteOffset = 24, InputSlotClass = D3D12_INPUT_CLASSIFICATION.D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new D3D12_INPUT_ELEMENT_DESC { SemanticName = "TEXCOORD", SemanticIndex = 0, Format = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_R32G32_FLOAT, InputSlot = 0, AlignedByteOffset = 28, InputSlotClass = D3D12_INPUT_CLASSIFICATION.D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new D3D12_INPUT_ELEMENT_DESC { SemanticName = "TANGENT", SemanticIndex = 0, Format =RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_FLOAT, InputSlot = 0, AlignedByteOffset = 36, InputSlotClass = D3D12_INPUT_CLASSIFICATION.D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
                new D3D12_INPUT_ELEMENT_DESC { SemanticName = "BINORMAL", SemanticIndex = 0, Format =RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_FLOAT, InputSlot = 0, AlignedByteOffset = 48, InputSlotClass = D3D12_INPUT_CLASSIFICATION.D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, InstanceDataStepRate = 0 },
            };
            d3d12GraphicsPipelineStateDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE.D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            d3d12GraphicsPipelineStateDesc.NumRenderTargets = 1;
            d3d12GraphicsPipelineStateDesc.RTVFormats0 = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM;
            d3d12GraphicsPipelineStateDesc.SampleDesc.Count = 1;
            d3d12GraphicsPipelineStateDesc.SampleDesc.Quality = 0;
            d3d12GraphicsPipelineState = d3d12Device.CreateGraphicsPipelineState(d3d12GraphicsPipelineStateDesc);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Command List.
            d3d12CommandList = d3d12Device.CreateCommandList(0U, D3D12_COMMAND_LIST_TYPE.D3D12_COMMAND_LIST_TYPE_DIRECT, d3d12CommandAllocator, d3d12GraphicsPipelineState);
        }
        void RenderDX()
        {
            if (wpfFrontBuffer == null || !IsVisible) return;
            ////////////////////////////////////////////////////////////////////////////////
            // Create Render Target View.
            var d3d12DescriptorHeap_Rtv = d3d12Device.CreateDescriptorHeap(new D3D12_DESCRIPTOR_HEAP_DESC { Type = D3D12_DESCRIPTOR_HEAP_TYPE.D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors = 1, Flags = D3D12_DESCRIPTOR_HEAP_FLAGS.D3D12_DESCRIPTOR_HEAP_FLAG_NONE });
            var d3d12RenderTargetViewDesc = new D3D12_RENDER_TARGET_VIEW_DESC { Format = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, ViewDimension = D3D12_RTV_DIMENSION.D3D12_RTV_DIMENSION_TEXTURE2D };
            d3d12RenderTargetViewDesc.__MIDL____MIDL_itf_RenderToy_0001_00180005.Texture2D = new D3D12_TEX2D_RTV { MipSlice = 0, PlaneSlice = 0 };
            var cpudesc = new D3D12_CPU_DESCRIPTOR_HANDLE();
            D3D12Shim.DescriptorHeap_GetCPUDescriptorHandleForHeapStart(d3d12DescriptorHeap_Rtv, ref cpudesc);
            d3d12Device.CreateRenderTargetView(d3d12Resource_RenderTarget, d3d12RenderTargetViewDesc, cpudesc);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Fence for end of frame.
            var d3d12Fence = d3d12Device.CreateFence(0, D3D12_FENCE_FLAGS.D3D12_FENCE_FLAG_NONE);
            var eventEndFrame = new AutoResetEvent(false);
            d3d12Fence.SetEventOnCompletion(1, eventEndFrame.GetSafeWaitHandle().DangerousGetHandle());
            ////////////////////////////////////////////////////////////////////////////////
            // Create a simple command list.
            unsafe
            {
                tagRECT *pRects = null;
                d3d12CommandList.ClearRenderTargetView(cpudesc, new float[] { 1, 0, 0, 1 }, 0, ref *pRects);
            }
            d3d12CommandList.SetGraphicsRootSignature(d3d12RootSignature);
            d3d12CommandList.SetPipelineState(d3d12GraphicsPipelineState);
            d3d12CommandList.IASetPrimitiveTopology(RenderToyCOM.D3D_PRIMITIVE_TOPOLOGY.D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            var scissor = new tagRECT { left = 0, top = 0, right = wpfFrontBuffer.PixelWidth, bottom = wpfFrontBuffer.PixelHeight };
            d3d12CommandList.RSSetScissorRects(1, scissor);
            var viewport = new D3D12_VIEWPORT { TopLeftX = 0, TopLeftY = 0, Width = wpfFrontBuffer.PixelWidth, Height = wpfFrontBuffer.PixelHeight, MinDepth = 0, MaxDepth = 1 };
            d3d12CommandList.RSSetViewports(1, viewport);
            unsafe
            {
                D3D12_CPU_DESCRIPTOR_HANDLE *handle = null;
                d3d12CommandList.OMSetRenderTargets(1, cpudesc, 0, ref *handle);
            }
            ////////////////////////////////////////////////////////////////////////////////
            // Create a vertex buffer to draw.
            var maintainBuffers = new List<ID3D12Resource>();
            var transformViewProjection = AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            foreach (var transformedobject in AttachedView.GetScene(this))
            {
                var transformModel = transformedobject.Transform;
                var transformModelViewProjection = transformModel * transformViewProjection;
                var vertexbuffer = CreateVertexBuffer(transformedobject.NodePrimitive);
                if (vertexbuffer == null) continue;
                d3d12CommandList.SetGraphicsRoot32BitConstants(0, 16, UnmanagedCopy.Create(Direct3DHelper.ConvertToD3DMatrix(transformModelViewProjection)), 0);
                var view = new D3D12_VERTEX_BUFFER_VIEW { BufferLocation = vertexbuffer.d3d12Resource_Buffer.GetGPUVirtualAddress(), SizeInBytes = vertexbuffer.size, StrideInBytes = (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) };
                d3d12CommandList.IASetVertexBuffers(0, 1, view);
                d3d12CommandList.DrawInstanced((uint)vertexbuffer.length, 1, 0, 0);
            }
            d3d12CommandList.Close();
            ////////////////////////////////////////////////////////////////////////////////
            // Create a command queue, submit the command list, and wait for completion.
            d3d12CommandQueue.ExecuteCommandLists(1, d3d12CommandList);
            d3d12CommandQueue.Signal(d3d12Fence, 1);
            eventEndFrame.WaitOne(100);
            d3d12CommandList.Reset(d3d12CommandAllocator, d3d12GraphicsPipelineState);
            ////////////////////////////////////////////////////////////////////////////////
            // Copy back the contents of the Render Target to WPF.
            wpfFrontBuffer.Lock();
            unsafe
            {
                D3D12_BOX* box = null;
                d3d12Resource_RenderTarget.ReadFromSubresource(wpfFrontBuffer.BackBuffer, (uint)wpfFrontBuffer.BackBufferStride, (uint)(wpfFrontBuffer.BackBufferStride * wpfFrontBuffer.PixelHeight), 0, ref *box);
            }
            wpfFrontBuffer.AddDirtyRect(new Int32Rect(0, 0, wpfFrontBuffer.PixelWidth, wpfFrontBuffer.PixelHeight));
            wpfFrontBuffer.Unlock();
        }
        byte[] ExtractRootSignature(byte[] shader)
        {
            var rootsignature = Direct3DCompiler.D3DGetBlobPart(shader, D3DBlobPart.RootSignature, 0);
            var buffer = rootsignature.GetBufferPointer();
            var buffersize = rootsignature.GetBufferSize();
            byte[] bytes = new byte[buffersize];
            Marshal.Copy(buffer, bytes, 0, (int)buffersize);
            return bytes;
        }
        class VertexBufferInfo
        {
            public ID3D12Resource d3d12Resource_Buffer;
            public int length;
            public uint size;
        }
        object Token = "DX12VertexBuffer";
        VertexBufferInfo CreateVertexBuffer(IPrimitive primitive)
        {
            if (primitive == null) return null;
            return MementoServer.Default.Get(primitive, Token, () =>
            {
                var verticesout = Direct3DHelper.ConvertToXYZNorDiffuseTex1(primitive);
                if (verticesout.Length == 0) return null;
                var size = (uint)(Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) * verticesout.Length);
                var d3d12ResourceDesc_Buffer = new D3D12_RESOURCE_DESC();
                d3d12ResourceDesc_Buffer.Dimension = D3D12_RESOURCE_DIMENSION.D3D12_RESOURCE_DIMENSION_BUFFER;
                d3d12ResourceDesc_Buffer.Width = size;
                d3d12ResourceDesc_Buffer.Height = 1;
                d3d12ResourceDesc_Buffer.DepthOrArraySize = 1;
                d3d12ResourceDesc_Buffer.MipLevels = 1;
                d3d12ResourceDesc_Buffer.SampleDesc.Count = 1;
                d3d12ResourceDesc_Buffer.Format = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
                d3d12ResourceDesc_Buffer.Layout = D3D12_TEXTURE_LAYOUT.D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                d3d12ResourceDesc_Buffer.Flags = D3D12_RESOURCE_FLAGS.D3D12_RESOURCE_FLAG_NONE;
                var d3d12HeapProperties_Upload = new D3D12_HEAP_PROPERTIES { Type = D3D12_HEAP_TYPE.D3D12_HEAP_TYPE_UPLOAD, CPUPageProperty = D3D12_CPU_PAGE_PROPERTY.D3D12_CPU_PAGE_PROPERTY_UNKNOWN, MemoryPoolPreference = D3D12_MEMORY_POOL.D3D12_MEMORY_POOL_UNKNOWN };
                var d3d12Resource_Buffer = d3d12Device.CreateCommittedResource(d3d12HeapProperties_Upload, D3D12_HEAP_FLAGS.D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES, d3d12ResourceDesc_Buffer, D3D12_RESOURCE_STATES.D3D12_RESOURCE_STATE_GENERIC_READ, null);
                IntPtr fillvertex = IntPtr.Zero;
                unsafe
                {
                    D3D12_RANGE* range = null;
                    d3d12Resource_Buffer.Map(0, ref *range, ref fillvertex);
                    var pin = GCHandle.Alloc(verticesout, GCHandleType.Pinned);
                    try
                    {
                        Buffer.MemoryCopy(Marshal.UnsafeAddrOfPinnedArrayElement(verticesout, 0).ToPointer(), fillvertex.ToPointer(), size, size);
                    }
                    finally
                    {
                        pin.Free();
                    }
                    d3d12Resource_Buffer.Unmap(0, ref *range);
                }
                return new VertexBufferInfo { d3d12Resource_Buffer = d3d12Resource_Buffer, length = verticesout.Length, size = size };
            });
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            wpfFrontBuffer = new WriteableBitmap((int)availableSize.Width, (int)availableSize.Height, 0, 0, PixelFormats.Bgra32, null);
            ////////////////////////////////////////////////////////////////////////////////
            // Create RenderTarget.
            var d3d12HeapProperties_Writeback = new D3D12_HEAP_PROPERTIES { Type = D3D12_HEAP_TYPE.D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty = D3D12_CPU_PAGE_PROPERTY.D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference = D3D12_MEMORY_POOL.D3D12_MEMORY_POOL_L0 };
            var d3d12ResourceDesc_RenderTarget = new D3D12_RESOURCE_DESC();
            d3d12ResourceDesc_RenderTarget.Dimension = D3D12_RESOURCE_DIMENSION.D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            d3d12ResourceDesc_RenderTarget.Width = (ulong)wpfFrontBuffer.PixelWidth;
            d3d12ResourceDesc_RenderTarget.Height = (uint)wpfFrontBuffer.PixelHeight;
            d3d12ResourceDesc_RenderTarget.DepthOrArraySize = 1;
            d3d12ResourceDesc_RenderTarget.MipLevels = 1;
            d3d12ResourceDesc_RenderTarget.SampleDesc.Count = 1;
            d3d12ResourceDesc_RenderTarget.Format = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM;
            d3d12ResourceDesc_RenderTarget.Layout = D3D12_TEXTURE_LAYOUT.D3D12_TEXTURE_LAYOUT_UNKNOWN;
            d3d12ResourceDesc_RenderTarget.Flags = D3D12_RESOURCE_FLAGS.D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            var d3d12ClearValue_RenderTarget = new D3D12ClearValue { Format = RenderToyCOM.DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, R = 1.0f, G = 0.0f, B = 0.0f, A = 1.0f };
            d3d12Resource_RenderTarget = d3d12Device.CreateCommittedResource(d3d12HeapProperties_Writeback, D3D12_HEAP_FLAGS.D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES, d3d12ResourceDesc_RenderTarget, D3D12_RESOURCE_STATES.D3D12_RESOURCE_STATE_RENDER_TARGET, d3d12ClearValue_RenderTarget);
            return base.MeasureOverride(availableSize);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            if (wpfFrontBuffer == null) return;
            RenderDX();
            drawingContext.DrawImage(wpfFrontBuffer, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        WriteableBitmap wpfFrontBuffer;
        ID3D12CommandQueue d3d12CommandQueue;
        ID3D12CommandAllocator d3d12CommandAllocator;
        ID3D12RootSignature d3d12RootSignature;
        ID3D12PipelineState d3d12GraphicsPipelineState;
        ID3D12GraphicsCommandList1 d3d12CommandList;
        ID3D12Resource d3d12Resource_RenderTarget;
    }
}