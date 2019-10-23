using System.Runtime.InteropServices;

namespace Arcturus.Managed
{
    class DrawingViewDXR12 : DrawingViewD3DImage
    {
        public DrawingViewDXR12()
        {
            m_renderTarget12 = Direct3D12.Device.OpenRenderTarget(m_renderTargetDeclaration, m_renderTarget9.GetIDirect3DSurface9Handle());
        }
        protected override void Update(FakeDocument document)
        {
            var vertexbuffer = Direct3D12.Device.CreateVertexBuffer((uint)(Marshal.SizeOf(typeof(Vertex)) * document.context.vertexCount()), (uint)(Marshal.SizeOf(typeof(Vertex))), document.context.vertexPointer());
            var indexbuffer = Direct3D12.Device.CreateIndexBuffer(sizeof(uint) * document.context.indexCount(), document.context.indexPointer());
            IDevice3D.TestRaytracer(Direct3D12.Device, m_renderTarget12, vertexbuffer, indexbuffer);
        }
        IRenderTarget m_renderTarget12;
    }
}