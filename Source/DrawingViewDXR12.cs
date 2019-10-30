using System.Runtime.InteropServices;

namespace Arcturus.Managed
{
    class DrawingViewDXR12 : DrawingViewD3DImage
    {
        public DrawingViewDXR12() : base(Direct3D12.Device, false)
        {
            m_renderTarget12 = m_device.OpenRenderTarget(m_renderTargetDeclaration, m_renderTarget9.GetIDirect3DSurface9Handle());
        }
        protected override IRenderTarget GetRenderTarget()
        {
            return m_renderTarget12;
        }
        protected override void Update(FakeDocument document)
        {
            var vertexbuffer = m_device.CreateVertexBuffer((uint)(Marshal.SizeOf(typeof(Vertex)) * document.context.vertexCount()), (uint)(Marshal.SizeOf(typeof(Vertex))), document.context.vertexPointer());
            var indexbuffer = m_device.CreateIndexBuffer(sizeof(uint) * document.context.indexCount(), document.context.indexPointer());
            IDevice3D.TestRaytracer(m_device, GetRenderTarget(), vertexbuffer, indexbuffer);
        }
        IRenderTarget m_renderTarget12;
    }
}