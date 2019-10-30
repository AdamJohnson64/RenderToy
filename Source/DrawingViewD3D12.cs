namespace Arcturus.Managed
{
    static class Direct3D12
    {
        public static IDevice3D Device = IDevice3D.CreateDevice3D_Direct3D12();
    }
    class DrawingViewD3D12 : DrawingViewD3DImage
    {
        public DrawingViewD3D12() : base(Direct3D12.Device, false)
        {
            m_renderTarget12 = Direct3D12.Device.OpenRenderTarget(m_renderTargetDeclaration, m_renderTarget9.GetIDirect3DSurface9Handle());
        }
        protected override IRenderTarget GetRenderTarget()
        {
            return m_renderTarget12;
        }
        IRenderTarget m_renderTarget12;
    }
}