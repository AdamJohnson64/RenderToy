using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class View3DDX12 : FrameworkElement
    {
        protected override void OnRender(DrawingContext drawingContext)
        {
            if (frontbuffer == null) return;
            ////////////////////////////////////////////////////////////////////////////////
            // Create the device if necessary.
            if (device == null)
            {
                Direct3D12.D3D12GetDebugInterface().EnableDebugLayer();
                device = Direct3D12.D3D12CreateDevice();
            }
            ////////////////////////////////////////////////////////////////////////////////
            // Create RenderTarget.
            var pHeapProperties = new D3D12HeapProperties { Type = D3D12HeapType.Custom, CPUPageProperty = D3D12CPUPageProperty.WriteBack, MemoryPoolPreference = D3D12MemoryPool.L0 };
            var pDesc = new D3D12ResourceDesc();
            pDesc.Dimension = D3D12ResourceDimension.Texture2D;
            pDesc.Width = (ulong)frontbuffer.PixelWidth;
            pDesc.Height = (uint)frontbuffer.PixelHeight;
            pDesc.DepthOrArraySize = 1;
            pDesc.SampleDesc.Count = 1;
            pDesc.Format = DXGIFormat.B8G8R8A8_Unorm;
            pDesc.Layout = D3D12TextureLayout.Unknown;
            pDesc.Flags = D3D12ResourceFlags.AllowRenderTarget;
            var pOptimizedClearValue = new D3D12ClearValue { Format = DXGIFormat.B8G8R8A8_Unorm, R = 1.0f, G = 0.0f, B = 0.0f, A = 1.0f };
            var rendertarget = device.CreateCommittedResource(pHeapProperties, D3D12HeapFlags.AllowAllBuffersAndTextures, pDesc, D3D12ResourceStates.RenderTarget, pOptimizedClearValue);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Pipeline State.
            commandAllocator = device.CreateCommandAllocator(D3D12CommandListType.Direct);
            var pipelinestatedesc = new D3D12GraphicsPipelineStateDesc();
            pipelinestatedesc.VS = CreateVertexShader();
            pipelinestatedesc.RasterizerState.CullMode = D3D12CullMode.None;
            pipelinestatedesc.RasterizerState.FillMode = D3D12FillMode.Solid;
            pipelinestatedesc.InputLayout.pInputElementDescs = new D3D12InputElementDesc[]
            {
                new D3D12InputElementDesc { SemanticName = "POSITION", SemanticIndex = 0, Format = DXGIFormat.R32G32B32_Float, InputSlot = 0, AlignedByteOffset = 0, InputSlotClass = D3D12InputClassification.PerVertexData, InstanceDataStepRate = 0 }
            };
            pipelinestatedesc.PrimitiveTopologyType = D3D12PrimitiveTopologyType.Triangle;
            var pipelinestate = device.CreateGraphicsPipelineState(pipelinestatedesc);
            ////////////////////////////////////////////////////////////////////////////////
            // Create Render Target View.
            var descriptorheap = device.CreateDescriptorHeap(new D3D12DescriptorHeapDesc { Type = D3D12DescriptorHeapType.Rtv, NumDescriptors = 1, Flags = D3D12DescriptorHeapFlags.None });
            var rtvdesc = new D3D12RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D12RtvDimension.Texture2D, Texture2D = new D3D12Tex2DRtv { MipSlice = 0, PlaneSlice = 0 } };
            device.CreateRenderTargetView(rendertarget, rtvdesc, descriptorheap.GetCPUDescriptorHandleForHeapStart());
            ////////////////////////////////////////////////////////////////////////////////
            // Create Fence for end of frame.
            var fence = device.CreateFence(0, D3D12FenceFlags.None);
            ////////////////////////////////////////////////////////////////////////////////
            // Create a simple command list.
            var commandlist = device.CreateCommandList(0U, D3D12CommandListType.Direct, commandAllocator, pipelinestate);
            commandlist.ClearRenderTargetView(descriptorheap.GetCPUDescriptorHandleForHeapStart(), 1.0f, 0.0f, 0.0f, 1.0f);
            commandlist.Close();
            ////////////////////////////////////////////////////////////////////////////////
            // Create a command queue, submit the command list, and wait for completion.
            var commandqueue = device.CreateCommandQueue(new D3D12CommandQueueDesc { Type = D3D12CommandListType.Direct });
            commandqueue.ExecuteCommandLists(new[] { commandlist });
            commandqueue.Signal(fence, 1);
            commandqueue.Wait(fence, 1);
            ////////////////////////////////////////////////////////////////////////////////
            // Copy back the contents of the Render Target to WPF.
            frontbuffer.Lock();
            rendertarget.ReadFromSubresource(frontbuffer.BackBuffer, (uint)frontbuffer.BackBufferStride, (uint)(frontbuffer.BackBufferStride * frontbuffer.PixelHeight), 0);
            frontbuffer.AddDirtyRect(new Int32Rect(0, 0, frontbuffer.PixelWidth, frontbuffer.PixelHeight));
            frontbuffer.Unlock();
            ////////////////////////////////////////////////////////////////////////////////
            // Draw the image.
            drawingContext.DrawImage(frontbuffer, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        byte[] CreateVertexShader()
        {
            if (cachedVS != null) return cachedVS;
            string vscode =
@"
#define CommonRoot \
""RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT),"" \
""RootConstants(num32BitConstants=16, b0, space=0, visibility=SHADER_VISIBILITY_ALL),"" \
""CBV(b16, space = 0, visibility=SHADER_VISIBILITY_ALL, flags=DATA_STATIC)""

float4x4 TransformModelViewProjection : register(c16);

struct VS_INPUT {
    float4 Position : POSITION;
};

struct VS_OUTPUT {
    float4 Position : SV_Position;
};

[RootSignature(CommonRoot)]
VS_OUTPUT vs(VS_INPUT input) {
    VS_OUTPUT result;
    result.Position = mul(TransformModelViewProjection, input.Position);
    return result;
}";
            D3DBlob code = new D3DBlob();
            D3DBlob error = new D3DBlob();
            Direct3DCompiler.D3DCompile(vscode, "temp.vs", "vs", "vs_5_0", 0, 0, code, error);
            var errorblobsize = error.GetBufferSize();
            var errorblob = error.GetBufferPointer();
            if (errorblobsize > 0 && errorblob != IntPtr.Zero)
            {
                var errors = Marshal.PtrToStringAnsi(errorblob, (int)errorblobsize - 1);
                throw new Exception("Shader compilation error:\n\n" + errors);
            }
            var buffer = code.GetBufferPointer();
            var buffersize = code.GetBufferSize();
            byte[] codebytes = new byte[buffersize];
            Marshal.Copy(buffer, codebytes, 0, (int)buffersize);
            cachedVS = codebytes;
            return codebytes;
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            frontbuffer = new WriteableBitmap((int)availableSize.Width, (int)availableSize.Height, 0, 0, PixelFormats.Bgra32, null);
            return base.MeasureOverride(availableSize);
        }
        D3D12Device device;
        D3D12CommandAllocator commandAllocator;
        static byte[] cachedVS;
        WriteableBitmap frontbuffer;
    }
}