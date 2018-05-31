using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewDirectX12 : FrameworkElement
    {
        protected override void OnRender(DrawingContext drawingContext)
        {
            if (wpfFrontBuffer == null) return;
            ////////////////////////////////////////////////////////////////////////////////
            // Create the device if necessary.
            if (d3d12Device == null)
            {
                Direct3D12.D3D12GetDebugInterface().EnableDebugLayer();
                d3d12Device = Direct3D12.D3D12CreateDevice();
            }
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
            var d3d12Resource_RenderTarget = d3d12Device.CreateCommittedResource(d3d12HeapProperties_Writeback, D3D12HeapFlags.AllowAllBuffersAndTextures, d3d12ResourceDesc_RenderTarget, D3D12ResourceStates.RenderTarget, d3d12ClearValue_RenderTarget);
            ////////////////////////////////////////////////////////////////////////////////
            // Create the Vertex Shader and Pixel Shader
            var bytecode_VertexShader = CompileVertexShader();
            var bytecode_PixelShader = CompilePixelShader();
            ////////////////////////////////////////////////////////////////////////////////
            // Extract the Root Signature
            byte[] bytecode_RootSignature = ExtractRootSignature(CompileVertexShader());
            var d3d12RootSignature = d3d12Device.CreateRootSignature(0, bytecode_RootSignature);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Pipeline State.
            d3d12CommandAllocator = d3d12Device.CreateCommandAllocator(D3D12CommandListType.Direct);
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
            var d3d12GraphicsPipelineState = d3d12Device.CreateGraphicsPipelineState(d3d12GraphicsPipelineStateDesc);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Render Target View.
            var d3d12DescriptorHeap_Rtv = d3d12Device.CreateDescriptorHeap(new D3D12DescriptorHeapDesc { Type = D3D12DescriptorHeapType.Rtv, NumDescriptors = 1, Flags = D3D12DescriptorHeapFlags.None });
            var d3d12RenderTargetViewDesc = new D3D12RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D12RtvDimension.Texture2D, Texture2D = new D3D12Tex2DRtv { MipSlice = 0, PlaneSlice = 0 } };
            d3d12Device.CreateRenderTargetView(d3d12Resource_RenderTarget, d3d12RenderTargetViewDesc, d3d12DescriptorHeap_Rtv.GetCPUDescriptorHandleForHeapStart());
            ////////////////////////////////////////////////////////////////////////////////
            // Create a vertex buffer to draw.
            var d3d12HeapProperties_Upload = new D3D12HeapProperties { Type = D3D12HeapType.Upload, CPUPageProperty = D3D12CPUPageProperty.Unknown, MemoryPoolPreference = D3D12MemoryPool.Unknown };
            var d3d12ResourceDesc_Buffer = new D3D12ResourceDesc();
            d3d12ResourceDesc_Buffer.Dimension = D3D12ResourceDimension.Buffer;
            d3d12ResourceDesc_Buffer.Width = 256;
            d3d12ResourceDesc_Buffer.Height = 1;
            d3d12ResourceDesc_Buffer.DepthOrArraySize = 1;
            d3d12ResourceDesc_Buffer.MipLevels = 1;
            d3d12ResourceDesc_Buffer.SampleDesc.Count = 1;
            d3d12ResourceDesc_Buffer.Format = DXGIFormat.Unknown;
            d3d12ResourceDesc_Buffer.Layout = D3D12TextureLayout.RowMajor;
            d3d12ResourceDesc_Buffer.Flags = D3D12ResourceFlags.None;
            var d3d12Resource_Buffer = d3d12Device.CreateCommittedResource(d3d12HeapProperties_Upload, D3D12HeapFlags.AllowAllBuffersAndTextures, d3d12ResourceDesc_Buffer, D3D12ResourceStates.GenericRead, null);
            IntPtr fillvertex = d3d12Resource_Buffer.Map(0);
            unsafe
            {
                float* fillme = (float*)fillvertex.ToPointer();
                fillme[0] = -1;
                fillme[1] = -1;
                fillme[2] = 0.5f;
                fillme[3] = 1;
                fillme[4] = -1;
                fillme[5] = 0.5f;
                fillme[6] = -1;
                fillme[7] = 1;
                fillme[8] = 0.5f;
            }
            d3d12Resource_Buffer.Unmap(0);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Fence for end of frame.
            var d3d12Fence = d3d12Device.CreateFence(0, D3D12FenceFlags.None);
            ////////////////////////////////////////////////////////////////////////////////
            // Create a simple command list.
            var d3d12CommandList = d3d12Device.CreateCommandList(0U, D3D12CommandListType.Direct, d3d12CommandAllocator, d3d12GraphicsPipelineState);
            d3d12CommandList.ClearRenderTargetView(d3d12DescriptorHeap_Rtv.GetCPUDescriptorHandleForHeapStart(), 1.0f, 0.0f, 0.0f, 1.0f);
            d3d12CommandList.SetGraphicsRootSignature(d3d12RootSignature);
            d3d12CommandList.SetPipelineState(d3d12GraphicsPipelineState);
            d3d12CommandList.IASetPrimitiveTopology(D3D12PrimitiveTopology.TriangleList);
            d3d12CommandList.IASetVertexBuffers(0, new[] { new D3D12VertexBufferView { BufferLocation = d3d12Resource_Buffer.GetGPUVirtualAddress(), SizeInBytes = 256, StrideInBytes = 12 } });
            d3d12CommandList.RSSetScissorRects(new[] { new D3D12Rect { left = 0, top = 0, right = wpfFrontBuffer.PixelWidth, bottom = wpfFrontBuffer.PixelHeight } });
            d3d12CommandList.RSSetViewports(new[] { new D3D12Viewport { TopLeftX = 0, TopLeftY = 0, Width = wpfFrontBuffer.PixelWidth, Height = wpfFrontBuffer.PixelHeight, MinDepth = 0, MaxDepth = 1 } });
            d3d12CommandList.OMSetRenderTargets(new[] { d3d12DescriptorHeap_Rtv.GetCPUDescriptorHandleForHeapStart() }, 0, null);
            d3d12CommandList.DrawInstanced(3, 1, 0, 0);
            d3d12CommandList.Close();
            ////////////////////////////////////////////////////////////////////////////////
            // Create a command queue, submit the command list, and wait for completion.
            var d3d12CommandQueue = d3d12Device.CreateCommandQueue(new D3D12CommandQueueDesc { Type = D3D12CommandListType.Direct });
            d3d12CommandQueue.ExecuteCommandLists(new[] { d3d12CommandList });
            d3d12CommandQueue.Signal(d3d12Fence, 1);
            d3d12CommandQueue.Wait(d3d12Fence, 1);
            ////////////////////////////////////////////////////////////////////////////////
            // Copy back the contents of the Render Target to WPF.
            wpfFrontBuffer.Lock();
            d3d12Resource_RenderTarget.ReadFromSubresource(wpfFrontBuffer.BackBuffer, (uint)wpfFrontBuffer.BackBufferStride, (uint)(wpfFrontBuffer.BackBufferStride * wpfFrontBuffer.PixelHeight), 0);
            wpfFrontBuffer.AddDirtyRect(new Int32Rect(0, 0, wpfFrontBuffer.PixelWidth, wpfFrontBuffer.PixelHeight));
            wpfFrontBuffer.Unlock();
            ////////////////////////////////////////////////////////////////////////////////
            // Draw the image.
            drawingContext.DrawImage(wpfFrontBuffer, new Rect(0, 0, ActualWidth, ActualHeight));
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

float4x4 TransformModelViewProjection : register(c16);

struct VS_INPUT {
    float3 Position : POSITION;
};

struct VS_OUTPUT {
    float4 Position : SV_Position;
};

[RootSignature(CommonRoot)]
VS_OUTPUT vs(VS_INPUT input) {
    VS_OUTPUT result;
    result.Position = float4(input.Position, 1); //mul(TransformModelViewProjection, input.Position);
    return result;
}

float4 ps(VS_OUTPUT input) : SV_Target {
    return float4(1, 1, 1, 1);
}
";
        }
        byte[] CompileVertexShader()
        {
            if (bytecode_CachedVS != null) return bytecode_CachedVS;
            return bytecode_CachedVS = CompileShader(GetShaderCode(), "vs", "vs_5_0");
        }
        byte[] CompilePixelShader()
        {
            if (bytecode_CachedPS != null) return bytecode_CachedPS;
            return bytecode_CachedPS = CompileShader(GetShaderCode(), "ps", "ps_5_0");
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
        protected override Size MeasureOverride(Size availableSize)
        {
            wpfFrontBuffer = new WriteableBitmap((int)availableSize.Width, (int)availableSize.Height, 0, 0, PixelFormats.Bgra32, null);
            return base.MeasureOverride(availableSize);
        }
        WriteableBitmap wpfFrontBuffer;
        D3D12Device d3d12Device;
        D3D12CommandAllocator d3d12CommandAllocator;
        static byte[] bytecode_CachedVS, bytecode_CachedPS;
    }
}