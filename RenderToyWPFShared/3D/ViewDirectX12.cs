using RenderToy.Cameras;
using RenderToy.PipelineModel;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewDirectX12 : FrameworkElement
    {
        public struct XYZ
        {
            public float Xp, Yp, Zp;
        };
        static ViewDirectX12()
        {
            AttachedView.SceneProperty.OverrideMetadata(typeof(ViewDirectX12), new FrameworkPropertyMetadata(null, (s, e) =>
            {
                ((ViewDirectX12)s).RenderDX();
            }));
            AttachedView.TransformModelViewProjectionProperty.OverrideMetadata(typeof(ViewDirectX12), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) =>
            {
                ((ViewDirectX12)s).RenderDX();
            }));
        }
        public ViewDirectX12()
        {
            Direct3D12.D3D12GetDebugInterface().EnableDebugLayer();
            d3d12Device = Direct3D12.D3D12CreateDevice();
            d3d12CommandQueue = d3d12Device.CreateCommandQueue(new D3D12CommandQueueDesc { Type = D3D12CommandListType.Direct });
            d3d12CommandAllocator = d3d12Device.CreateCommandAllocator(D3D12CommandListType.Direct);
            ////////////////////////////////////////////////////////////////////////////////
            // Create the Vertex Shader and Pixel Shader
            var bytecode_VertexShader = CompileShader(GetShaderCode(), "vs", "vs_5_0");
            var bytecode_PixelShader = CompileShader(GetShaderCode(), "ps", "ps_5_0");
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
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.BlendEnable = 0;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.SrcBlend = D3D12Blend.One;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.DestBlend = D3D12Blend.Zero;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.BlendOp = D3D12BlendOp.Add;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.SrcBlendAlpha = D3D12Blend.One;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.DestBlendAlpha = D3D12Blend.Zero;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.BlendOpAlpha = D3D12BlendOp.Add;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.LogicOp = D3D12LogicOp.NoOp;
            d3d12GraphicsPipelineStateDesc.BlendState.RenderTarget0.RenderTargetWriteMask = 15;
            d3d12GraphicsPipelineStateDesc.SampleMask = uint.MaxValue;
            d3d12GraphicsPipelineStateDesc.RasterizerState.FillMode = D3D12FillMode.Solid;
            d3d12GraphicsPipelineStateDesc.RasterizerState.CullMode = D3D12CullMode.None;
            d3d12GraphicsPipelineStateDesc.InputLayout.pInputElementDescs = new D3D12InputElementDesc[]
            {
                new D3D12InputElementDesc { SemanticName = "POSITION", SemanticIndex = 0, Format = DXGIFormat.R32G32B32_Float, InputSlot = 0, AlignedByteOffset = 0, InputSlotClass = D3D12InputClassification.PerVertexData, InstanceDataStepRate = 0 }
            };
            d3d12GraphicsPipelineStateDesc.PrimitiveTopologyType = D3D12PrimitiveTopologyType.Triangle;
            d3d12GraphicsPipelineStateDesc.NumRenderTargets = 1;
            d3d12GraphicsPipelineStateDesc.RTVFormats0 = DXGIFormat.B8G8R8A8_Unorm;
            d3d12GraphicsPipelineStateDesc.SampleDesc.Count = 1;
            d3d12GraphicsPipelineStateDesc.SampleDesc.Quality = 0;
            d3d12GraphicsPipelineState = d3d12Device.CreateGraphicsPipelineState(d3d12GraphicsPipelineStateDesc);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Command List.
            d3d12CommandList = d3d12Device.CreateCommandList(0U, D3D12CommandListType.Direct, d3d12CommandAllocator, d3d12GraphicsPipelineState);
        }
        void RenderDX()
        {
            if (wpfFrontBuffer == null) return;
            ////////////////////////////////////////////////////////////////////////////////
            // Create Render Target View.
            var d3d12DescriptorHeap_Rtv = d3d12Device.CreateDescriptorHeap(new D3D12DescriptorHeapDesc { Type = D3D12DescriptorHeapType.Rtv, NumDescriptors = 1, Flags = D3D12DescriptorHeapFlags.None });
            var d3d12RenderTargetViewDesc = new D3D12RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D12RtvDimension.Texture2D, Texture2D = new D3D12Tex2DRtv { MipSlice = 0, PlaneSlice = 0 } };
            d3d12Device.CreateRenderTargetView(d3d12Resource_RenderTarget, d3d12RenderTargetViewDesc, d3d12DescriptorHeap_Rtv.GetCPUDescriptorHandleForHeapStart());
            ////////////////////////////////////////////////////////////////////////////////
            // Create Fence for end of frame.
            var d3d12Fence = d3d12Device.CreateFence(0, D3D12FenceFlags.None);
            var eventEndFrame = new AutoResetEvent(false);
            d3d12Fence.SetEventOnCompletion(1, eventEndFrame.GetSafeWaitHandle().DangerousGetHandle());
            ////////////////////////////////////////////////////////////////////////////////
            // Create a simple command list.
            d3d12CommandList.ClearRenderTargetView(d3d12DescriptorHeap_Rtv.GetCPUDescriptorHandleForHeapStart(), 1.0f, 0.0f, 0.0f, 1.0f);
            d3d12CommandList.SetGraphicsRootSignature(d3d12RootSignature);
            d3d12CommandList.SetPipelineState(d3d12GraphicsPipelineState);
            d3d12CommandList.IASetPrimitiveTopology(D3D12PrimitiveTopology.TriangleList);
            d3d12CommandList.RSSetScissorRects(new[] { new D3D12Rect { left = 0, top = 0, right = wpfFrontBuffer.PixelWidth, bottom = wpfFrontBuffer.PixelHeight } });
            d3d12CommandList.RSSetViewports(new[] { new D3D12Viewport { TopLeftX = 0, TopLeftY = 0, Width = wpfFrontBuffer.PixelWidth, Height = wpfFrontBuffer.PixelHeight, MinDepth = 0, MaxDepth = 1 } });
            d3d12CommandList.OMSetRenderTargets(new[] { d3d12DescriptorHeap_Rtv.GetCPUDescriptorHandleForHeapStart() }, 0, null);
            ////////////////////////////////////////////////////////////////////////////////
            // Create a vertex buffer to draw.
            var maintainBuffers = new List<D3D12Resource>();
            var transformViewProjection = AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            foreach (var transformedobject in TransformedObject.Enumerate(AttachedView.GetScene(this)))
            {
                var transformModel = transformedobject.Transform;
                var transformModelViewProjection = transformModel * transformViewProjection;
                var vertexbuffer = CreateVertexBuffer(transformedobject.Node.Primitive);
                if (vertexbuffer == null) continue;
                float[] mvp = new float[16]
                {
                    (float)transformModelViewProjection.M11, (float)transformModelViewProjection.M12, (float)transformModelViewProjection.M13, (float)transformModelViewProjection.M14,
                    (float)transformModelViewProjection.M21, (float)transformModelViewProjection.M22, (float)transformModelViewProjection.M23, (float)transformModelViewProjection.M24,
                    (float)transformModelViewProjection.M31, (float)transformModelViewProjection.M32, (float)transformModelViewProjection.M33, (float)transformModelViewProjection.M34,
                    (float)transformModelViewProjection.M41, (float)transformModelViewProjection.M42, (float)transformModelViewProjection.M43, (float)transformModelViewProjection.M44,
                };
                d3d12CommandList.SetGraphicsRoot32BitConstants(0, 16, mvp, 0);
                d3d12CommandList.IASetVertexBuffers(0, new[] { new D3D12VertexBufferView { BufferLocation = vertexbuffer.d3d12Resource_Buffer.GetGPUVirtualAddress(), SizeInBytes = vertexbuffer.size, StrideInBytes = 12 } });
                d3d12CommandList.DrawInstanced((uint)vertexbuffer.length, 1, 0, 0);
            }
            d3d12CommandList.Close();
            ////////////////////////////////////////////////////////////////////////////////
            // Create a command queue, submit the command list, and wait for completion.
            d3d12CommandQueue.ExecuteCommandLists(new[] { d3d12CommandList });
            d3d12CommandQueue.Signal(d3d12Fence, 1);
            eventEndFrame.WaitOne(100);
            d3d12CommandList.Reset(d3d12CommandAllocator, d3d12GraphicsPipelineState);
            ////////////////////////////////////////////////////////////////////////////////
            // Copy back the contents of the Render Target to WPF.
            wpfFrontBuffer.Lock();
            d3d12Resource_RenderTarget.ReadFromSubresource(wpfFrontBuffer.BackBuffer, (uint)wpfFrontBuffer.BackBufferStride, (uint)(wpfFrontBuffer.BackBufferStride * wpfFrontBuffer.PixelHeight), 0);
            wpfFrontBuffer.AddDirtyRect(new Int32Rect(0, 0, wpfFrontBuffer.PixelWidth, wpfFrontBuffer.PixelHeight));
            wpfFrontBuffer.Unlock();
        }
        byte[] CompileShader(string shader, string entrypoint, string profile)
        {
            D3DBlob code = new D3DBlob();
            D3DBlob error = new D3DBlob();
            Direct3DCompiler.D3DCompile(shader, "temp", entrypoint, profile, 0, 0, code, error);
            var errorblobsize = error.GetBufferSize();
            var errorblob = error.GetBufferPointer();
            if (errorblobsize > 0 && errorblob != IntPtr.Zero)
            {
                var errors = Marshal.PtrToStringAnsi(errorblob, (int)errorblobsize - 1);
                throw new Exception("Shader compilation error:\n\n" + errors);
            }
            var buffer = code.GetBufferPointer();
            var buffersize = code.GetBufferSize();
            byte[] bytecode = new byte[buffersize];
            Marshal.Copy(buffer, bytecode, 0, (int)buffersize);
            return bytecode;
        }
        string GetShaderCode()
        {
            return
@"#define CommonRoot \
""RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT),"" \
""RootConstants(num32BitConstants=16, b0, space=0, visibility=SHADER_VISIBILITY_ALL)""

cbuffer Constants : register(b0)
{
    float4x4 TransformModelViewProjection;
};

struct VS_INPUT {
    float3 Position : POSITION;
};

struct VS_OUTPUT {
    float4 Position : SV_Position;
};

[RootSignature(CommonRoot)]
VS_OUTPUT vs(VS_INPUT input) {
    VS_OUTPUT result;
    //result.Position = float4(input.Position, 1);
    result.Position = mul(TransformModelViewProjection, float4(input.Position, 1));
    return result;
}

float4 ps(VS_OUTPUT input) : SV_Target {
    return float4(1, 1, 1, 1);
}
";
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
            public D3D12Resource d3d12Resource_Buffer;
            public int length;
            public uint size;
        }
        object Token = "";
        VertexBufferInfo CreateVertexBuffer(IPrimitive primitive)
        {
            if (primitive == null) return null;
            return MementoServer.Default.Get(primitive, Token, () =>
            {
                var verticesin = PrimitiveAssembly.CreateTrianglesDX(primitive);
                var verticesout = verticesin.Select(i => new XYZ { Xp = (float)i.Position.X, Yp = (float)i.Position.Y, Zp = (float)i.Position.Z }).ToArray();
                if (verticesout.Length == 0) return null;
                var size = (uint)(Marshal.SizeOf(typeof(XYZ)) * verticesout.Length);
                var d3d12ResourceDesc_Buffer = new D3D12ResourceDesc();
                d3d12ResourceDesc_Buffer.Dimension = D3D12ResourceDimension.Buffer;
                d3d12ResourceDesc_Buffer.Width = size;
                d3d12ResourceDesc_Buffer.Height = 1;
                d3d12ResourceDesc_Buffer.DepthOrArraySize = 1;
                d3d12ResourceDesc_Buffer.MipLevels = 1;
                d3d12ResourceDesc_Buffer.SampleDesc.Count = 1;
                d3d12ResourceDesc_Buffer.Format = DXGIFormat.Unknown;
                d3d12ResourceDesc_Buffer.Layout = D3D12TextureLayout.RowMajor;
                d3d12ResourceDesc_Buffer.Flags = D3D12ResourceFlags.None;
                var d3d12HeapProperties_Upload = new D3D12HeapProperties { Type = D3D12HeapType.Upload, CPUPageProperty = D3D12CPUPageProperty.Unknown, MemoryPoolPreference = D3D12MemoryPool.Unknown };
                var d3d12Resource_Buffer = d3d12Device.CreateCommittedResource(d3d12HeapProperties_Upload, D3D12HeapFlags.AllowAllBuffersAndTextures, d3d12ResourceDesc_Buffer, D3D12ResourceStates.GenericRead, null);
                IntPtr fillvertex = d3d12Resource_Buffer.Map(0);
                unsafe
                {
                    Buffer.MemoryCopy(Marshal.UnsafeAddrOfPinnedArrayElement(verticesout, 0).ToPointer(), fillvertex.ToPointer(), size, size);
                }
                d3d12Resource_Buffer.Unmap(0);
                return new VertexBufferInfo { d3d12Resource_Buffer = d3d12Resource_Buffer, length = verticesout.Length, size = size };
            });
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            wpfFrontBuffer = new WriteableBitmap((int)availableSize.Width, (int)availableSize.Height, 0, 0, PixelFormats.Bgra32, null);
            ////////////////////////////////////////////////////////////////////////////////
            // Create RenderTarget.
            var d3d12HeapProperties_Writeback = new D3D12HeapProperties { Type = D3D12HeapType.Custom, CPUPageProperty = D3D12CPUPageProperty.WriteBack, MemoryPoolPreference = D3D12MemoryPool.L0 };
            var d3d12ResourceDesc_RenderTarget = new D3D12ResourceDesc();
            d3d12ResourceDesc_RenderTarget.Dimension = D3D12ResourceDimension.Texture2D;
            d3d12ResourceDesc_RenderTarget.Width = (ulong)wpfFrontBuffer.PixelWidth;
            d3d12ResourceDesc_RenderTarget.Height = (uint)wpfFrontBuffer.PixelHeight;
            d3d12ResourceDesc_RenderTarget.DepthOrArraySize = 1;
            d3d12ResourceDesc_RenderTarget.SampleDesc.Count = 1;
            d3d12ResourceDesc_RenderTarget.Format = DXGIFormat.B8G8R8A8_Unorm;
            d3d12ResourceDesc_RenderTarget.Layout = D3D12TextureLayout.Unknown;
            d3d12ResourceDesc_RenderTarget.Flags = D3D12ResourceFlags.AllowRenderTarget;
            var d3d12ClearValue_RenderTarget = new D3D12ClearValue { Format = DXGIFormat.B8G8R8A8_Unorm, R = 1.0f, G = 0.0f, B = 0.0f, A = 1.0f };
            d3d12Resource_RenderTarget = d3d12Device.CreateCommittedResource(d3d12HeapProperties_Writeback, D3D12HeapFlags.AllowAllBuffersAndTextures, d3d12ResourceDesc_RenderTarget, D3D12ResourceStates.RenderTarget, d3d12ClearValue_RenderTarget);
            ////////////////////////////////////////////////////////////////////////////////
            // Render.
            RenderDX();
            RenderDX();
            return base.MeasureOverride(availableSize);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(wpfFrontBuffer, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        WriteableBitmap wpfFrontBuffer;
        D3D12Device d3d12Device;
        D3D12CommandQueue d3d12CommandQueue;
        D3D12CommandAllocator d3d12CommandAllocator;
        D3D12RootSignature d3d12RootSignature;
        D3D12PipelineState d3d12GraphicsPipelineState;
        D3D12GraphicsCommandList1 d3d12CommandList;
        D3D12Resource d3d12Resource_RenderTarget;
    }
}