using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewDirectX11 : FrameworkElement
    {
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
        }
        public ViewDirectX11()
        {
            var hlsl =
@"cbuffer Constants : register(b0)
{
    float4x4 TransformModelViewProjection;
};

struct VS_INPUT {
    float3 Position : POSITION;
};

struct VS_OUTPUT {
    float4 Position : SV_Position;
};

VS_OUTPUT vs(VS_INPUT input) {
    VS_OUTPUT result;
    //result.Position = float4(input.Position, 1);
    result.Position = mul(TransformModelViewProjection, float4(input.Position, 1));
    return result;
}

float4 ps(VS_OUTPUT input) : SV_Target {
    return float4(1, 1, 1, 1);
}";
            d3d11Device = Direct3D11.D3D11CreateDevice();
            {
                var bytecode = HLSLExtension.CompileHLSL(hlsl, "vs", "vs_5_0");
                d3d11InputLayout = d3d11Device.CreateInputLayout(new[] { new D3D11InputElementDesc { SemanticName = "POSITION", SemanticIndex = 0, Format = DXGIFormat.R32G32B32_Float, InputSlot = 0, AlignedByteOffset = 0, InputSlotClass = D3D11InputClassification.PerVertexData, InstanceDataStepRate = 0 }, }, bytecode);
                d3d11VertexShader = d3d11Device.CreateVertexShader(bytecode);
            }
            {
                var bytecode = HLSLExtension.CompileHLSL(hlsl, "ps", "ps_5_0");
                d3d11PixelShader = d3d11Device.CreatePixelShader(bytecode);
            }
            d3d11RasterizerState = d3d11Device.CreateRasterizerState(new D3D11RasterizerDesc { FillMode = D3D11FillMode.Solid, CullMode = D3D11CullMode.None });
        }
        void RenderDX()
        {
            if (wpfFrontBuffer == null || !IsVisible) return;
            var context = d3d11Device.GetImmediateContext();
            context.ClearRenderTargetView(d3d11RenderTargetView, 1, 0, 0, 1);
            context.IASetPrimitiveTopology(D3DPrimitiveTopology.TriangleList);
            context.IASetInputLayout(d3d11InputLayout);
            context.VSSetShader(d3d11VertexShader);
            context.PSSetShader(d3d11PixelShader);
            context.RSSetState(d3d11RasterizerState);
            context.RSSetScissorRects(new[] { new D3D11Rect { left = 0, top = 0, right = wpfFrontBuffer.PixelWidth, bottom = wpfFrontBuffer.PixelHeight } });
            context.RSSetViewports(new[] { new D3D11Viewport { TopLeftX = 0, TopLeftY = 0, Width = wpfFrontBuffer.PixelWidth, Height = wpfFrontBuffer.PixelHeight, MinDepth = 0, MaxDepth = 1 } });
            context.OMSetRenderTargets(new[] { d3d11RenderTargetView });
            ////////////////////////////////////////////////////////////////////////////////
            // Draw the scene.
            var transformViewProjection = AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            foreach (var transformedobject in TransformedObject.Enumerate(AttachedView.GetScene(this)))
            {
                var transformModel = transformedobject.Transform;
                var transformModelViewProjection = transformModel * transformViewProjection;
                var vertexbuffer = CreateVertexBuffer(transformedobject.Node.Primitive);
                if (vertexbuffer == null) continue;
                var d3d11ConstantBuffer = d3d11Device.CreateBuffer(new D3D11BufferDesc { ByteWidth = 4 * 16, Usage = D3D11Usage.Immutable, BindFlags = D3D11BindFlag.ConstantBuffer, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = 4 * 16}, new D3D11SubresourceData { pSysMem = DirectXHelper.ConvertToD3DMatrix(transformModelViewProjection), SysMemPitch = 0, SysMemSlicePitch = 0 });
                context.VSSetConstantBuffers(0, new[] { d3d11ConstantBuffer });
                context.IASetVertexBuffers(0, new[] { vertexbuffer.d3d11Buffer }, new[] { (uint)Marshal.SizeOf(typeof(XYZ)) }, new[] { 0U });
                context.Draw(vertexbuffer.vertexCount, 0);
            }
            ////////////////////////////////////////////////////////////////////////////////
            // Copy back the Render Target.
            context.CopyResource(d3d11Texture2D_Copyback, d3d11Texture2D_RT);
            context.Flush();
            var d3d11Map = context.Map(d3d11Texture2D_Copyback, 0, D3D11Map.Read, 0);
            int bitmapWidth = wpfFrontBuffer.PixelWidth;
            int bitmapHeight = wpfFrontBuffer.PixelHeight;
            int bitmapStride = wpfFrontBuffer.BackBufferStride;
            wpfFrontBuffer.Lock();
            unsafe
            {
                void* pBitmapIn = d3d11Map.pData.ToPointer();
                void* pBitmapOut = wpfFrontBuffer.BackBuffer.ToPointer();
                for (int y = 0; y < bitmapHeight; ++y)
                {
                    uint* pRasterIn = (uint*)((byte*)pBitmapIn + d3d11Map.RowPitch * y); 
                    uint* pRasterOut = (uint*)((byte*)pBitmapOut + bitmapStride * y);
                    for (int x = 0; x < bitmapWidth; ++x)
                    {
                        pRasterOut[x] = pRasterIn[x];
                    }
                }
            }
            wpfFrontBuffer.AddDirtyRect(new Int32Rect(0, 0, wpfFrontBuffer.PixelWidth, wpfFrontBuffer.PixelHeight));
            wpfFrontBuffer.Unlock();
            context.Unmap(d3d11Texture2D_Copyback, 0);
        }
        class VertexBufferInfo
        {
            public D3D11Buffer d3d11Buffer;
            public uint vertexCount;
        }
        object Token = "DX11VertexBuffer";
        VertexBufferInfo CreateVertexBuffer(IPrimitive primitive)
        {
            if (primitive == null) return null;
            return MementoServer.Default.Get(primitive, Token, () =>
            {
                var verticesout = DirectXHelper.ConvertToXYZ(primitive);
                if (verticesout.Length == 0) return null;
                var size = (uint)(Marshal.SizeOf(typeof(XYZ)) * verticesout.Length);
                var d3d11Buffer = d3d11Device.CreateBuffer(
                    new D3D11BufferDesc { ByteWidth = size, Usage = D3D11Usage.Immutable, BindFlags = D3D11BindFlag.VertexBuffer, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = (uint)Marshal.SizeOf(typeof(XYZ)) },
                    new D3D11SubresourceData { pSysMem = verticesout, SysMemPitch = 0, SysMemSlicePitch = 0 });
                return new VertexBufferInfo { d3d11Buffer = d3d11Buffer, vertexCount = (uint)verticesout.Length };
            });
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            wpfFrontBuffer = new WriteableBitmap((int)availableSize.Width, (int)availableSize.Height, 0, 0, PixelFormats.Bgra32, null);
            var d3d11Texture2DDesc_RT = new D3D11Texture2DDesc { Width = (uint)wpfFrontBuffer.PixelWidth, Height = (uint)wpfFrontBuffer.PixelHeight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.B8G8R8A8_Unorm, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.RenderTarget, CPUAccessFlags = 0 };
            d3d11Texture2D_RT = d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT, null);
            d3d11RenderTargetView = d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });
            var d3d11Texture2DDesc_Copyback = new D3D11Texture2DDesc { Width = (uint)wpfFrontBuffer.PixelWidth, Height = (uint)wpfFrontBuffer.PixelHeight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.B8G8R8A8_Unorm, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Staging, BindFlags = 0, CPUAccessFlags = D3D11CpuAccessFlag.Read };
            d3d11Texture2D_Copyback = d3d11Device.CreateTexture2D(d3d11Texture2DDesc_Copyback, null);
            return base.MeasureOverride(availableSize);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            if (wpfFrontBuffer == null) return;
            RenderDX();
            drawingContext.DrawImage(wpfFrontBuffer, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        WriteableBitmap wpfFrontBuffer;
        D3D11Device d3d11Device;
        D3D11InputLayout d3d11InputLayout;
        D3D11RasterizerState d3d11RasterizerState;
        D3D11VertexShader d3d11VertexShader;
        D3D11PixelShader d3d11PixelShader;
        D3D11Texture2D d3d11Texture2D_RT;
        D3D11RenderTargetView d3d11RenderTargetView;
        D3D11Texture2D d3d11Texture2D_Copyback;
    }
}