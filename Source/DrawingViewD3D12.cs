using System;
using System.Runtime.InteropServices;

namespace Arcturus.Managed
{
    static class Direct3D12
    {
        public static IDevice3D Device = IDevice3D.CreateDevice3D_Direct3D12();
    }
    class DrawingViewD3D12 : DrawingViewD3DImage
    {
        public DrawingViewD3D12()
        {
            m_shader12 = Direct3D12.Device.CreateShader();
            m_renderTarget12 = Direct3D12.Device.OpenRenderTarget(m_renderTargetDeclaration, m_renderTarget9.GetIDirect3DSurface9Handle());
        }
        protected override void Update(FakeDocument document)
        {
            var vertexbuffer = Direct3D12.Device.CreateVertexBuffer((uint)(Marshal.SizeOf(typeof(Vertex)) * document.context.vertexCount()), (uint)(Marshal.SizeOf(typeof(Vertex))), document.context.vertexPointer());
            var indexbuffer = Direct3D12.Device.CreateIndexBuffer(sizeof(uint) * document.context.indexCount(), document.context.indexPointer());
            IConstantBuffer constantBuffer12;
            unsafe
            {
                Matrix vertexTransform;
                vertexTransform.M11 = 2.0f / 256.0f;
                vertexTransform.M22 = -2.0f / 256.0f;
                vertexTransform.M33 = 1;
                vertexTransform.M41 = -1;
                vertexTransform.M42 = 1;
                vertexTransform.M44 = 1;
                constantBuffer12 = Direct3D12.Device.CreateConstantBuffer((uint)Marshal.SizeOf(typeof(Matrix)), new IntPtr(&vertexTransform));
            }
            var constantBufferView12 = Direct3D12.Device.CreateConstantBufferView(constantBuffer12);
            Direct3D12.Device.BeginRender();
            Direct3D12.Device.BeginPass(m_renderTarget12, new Color());
            Direct3D12.Device.SetViewport(new Viewport { width = 256, height = 256, maxDepth = 1 });
            Direct3D12.Device.SetShader(m_shader12);
            Direct3D12.Device.SetVertexBuffer(vertexbuffer, (uint)Marshal.SizeOf(typeof(Vertex)));
            Direct3D12.Device.SetIndexBuffer(indexbuffer);
            Direct3D12.Device.DrawIndexedPrimitives(document.context.vertexCount(), document.context.indexCount());
            Direct3D12.Device.EndPass();
            Direct3D12.Device.EndRender();
        }
        IShader m_shader12;
        IRenderTarget m_renderTarget12;
    }
}