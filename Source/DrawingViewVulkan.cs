using System;
using System.Runtime.InteropServices;

namespace Arcturus.Managed
{
    static class Vulkan
    {
        public static IDevice3D Device = IDevice3D.CreateDevice3D_Vulkan();
    }
    class DrawingViewVulkan : DrawingViewD3DImage
    {
        public DrawingViewVulkan()
        {
            m_backBuffer12 = Direct3D12.Device.OpenRenderTarget(m_renderTargetDeclaration, m_renderTarget9.GetIDirect3DSurface9Handle());
            m_shaderVK = Vulkan.Device.CreateShader();
            m_renderTargetVK = Vulkan.Device.OpenRenderTarget(m_renderTargetDeclaration, IDevice3D.GetID3D12Texture2DHandleNT(m_renderTarget12));
        }
        protected override void Update(FakeDocument document)
        {
            var vertexbuffer = Vulkan.Device.CreateVertexBuffer((uint)(Marshal.SizeOf(typeof(Vertex)) * document.context.vertexCount()), (uint)(Marshal.SizeOf(typeof(Vertex))), document.context.vertexPointer());
            var indexbuffer = Vulkan.Device.CreateIndexBuffer(sizeof(uint) * document.context.indexCount(), document.context.indexPointer());
            IConstantBuffer constantBufferVK;
            unsafe
            {
                Matrix vertexTransform;
                vertexTransform.M11 = 2.0f / 256.0f;
                vertexTransform.M22 = 2.0f / 256.0f;
                vertexTransform.M33 = 1;
                vertexTransform.M41 = -1;
                vertexTransform.M42 = -1;
                vertexTransform.M44 = 1;
                constantBufferVK = Vulkan.Device.CreateConstantBuffer((uint)Marshal.SizeOf(typeof(Matrix)), new IntPtr(&vertexTransform));
            }
            var constantBufferViewVK = Vulkan.Device.CreateConstantBufferView(constantBufferVK);
            Vulkan.Device.BeginRender();
            Vulkan.Device.BeginPass(m_renderTargetVK, new Color());
            Vulkan.Device.SetViewport(new Viewport { width = 256, height = 256, maxDepth = 1 });
            Vulkan.Device.SetShader(m_shaderVK);
            Vulkan.Device.SetVertexBuffer(vertexbuffer, (uint)Marshal.SizeOf(typeof(Vertex)));
            Vulkan.Device.SetIndexBuffer(indexbuffer);
            Vulkan.Device.DrawIndexedPrimitives(document.context.vertexCount(), document.context.indexCount());
            Vulkan.Device.EndPass();
            Vulkan.Device.EndRender();
            Direct3D12.Device.CopyResource(m_backBuffer12, m_renderTarget12);
        }
        IShader m_shaderVK;
        IRenderTarget m_backBuffer12;
        IRenderTarget m_renderTarget12 = Direct3D12.Device.CreateRenderTarget(new RenderTargetDeclaration { width = 256, height = 256 });
        IRenderTarget m_renderTargetVK;
    }
}