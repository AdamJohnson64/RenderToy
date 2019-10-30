namespace Arcturus.Managed
{
    static class Vulkan
    {
        public static IDevice3D Device = IDevice3D.CreateDevice3D_Vulkan();
    }
    class DrawingViewVulkan : DrawingViewD3DImage
    {
        public DrawingViewVulkan() : base(Vulkan.Device, true)
        {
            m_backBuffer12 = Direct3D12.Device.OpenRenderTarget(m_renderTargetDeclaration, m_renderTarget9.GetIDirect3DSurface9Handle());
            m_renderTarget12 = Direct3D12.Device.CreateRenderTarget(new RenderTargetDeclaration { width = 256, height = 256 });
            m_renderTargetVK = m_device.OpenRenderTarget(m_renderTargetDeclaration, IDevice3D.GetID3D12Texture2DHandleNT(m_renderTarget12));
        }
        protected override IRenderTarget GetRenderTarget()
        {
            return m_renderTargetVK;
        }
        protected override void Update(FakeDocument document)
        {
            base.Update(document);
            Direct3D12.Device.CopyResource(m_backBuffer12, m_renderTarget12);
        }
        IRenderTarget m_backBuffer12;
        IRenderTarget m_renderTarget12;
        IRenderTarget m_renderTargetVK;
    }
}