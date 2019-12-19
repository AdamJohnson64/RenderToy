using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Threading;

namespace Arcturus.Managed
{
    static class Direct3D9
    {
        public static IDevice3D_D3D9 Device = new IDevice3D_D3D9();
    }
    public abstract class DrawingViewD3DImage : ContentControl
    {
        static DrawingViewD3DImage()
        {
            DataContextProperty.OverrideMetadata(typeof(DrawingViewD3DImage), new FrameworkPropertyMetadata(DocumentChanged));
        }
        protected DrawingViewD3DImage(IDevice3D device, bool flip)
        {
            m_device = device;
            m_shader = m_device.CreateShader();
            UInt32[] pixels = new UInt32[256 * 256];
            for (uint y = 0; y < 256; ++y)
            {
                for (uint x = 0; x < 256; ++x)
                {
                    pixels[x + y * 256] = (x << 16) | (y << 8) | 0xFF000000U;
                }
            }
            unsafe
            {
                fixed (UInt32* pPixels = pixels)
                {
                    m_texture = m_device.CreateTexture2D(256, 256, new IntPtr(pPixels));
                }
            }
            unsafe
            {
                Matrix vertexTransform;
                vertexTransform.M11 = 2.0f / 256.0f;
                vertexTransform.M22 = (flip ? -1 : 1) * -2.0f / 256.0f;
                vertexTransform.M33 = 1;
                vertexTransform.M41 = -1;
                vertexTransform.M42 = flip ? -1 : 1;
                vertexTransform.M44 = 1;
                m_constantBuffer = m_device.CreateConstantBuffer((uint)Marshal.SizeOf(typeof(Matrix)), new IntPtr(&vertexTransform));
            }
            m_constantBufferView = m_device.CreateConstantBufferView(m_constantBuffer);
            var grid = new Grid();
            var image = new Image { Source = m_d3dImage, HorizontalAlignment = HorizontalAlignment.Left, VerticalAlignment = VerticalAlignment.Top, Width = 256, Height = 256 };
            RenderOptions.SetBitmapScalingMode(image, BitmapScalingMode.NearestNeighbor);
            m_timer.Interval = TimeSpan.FromSeconds(1);
            m_timer.Tick += (s, e) => Update();
            m_timer.Start();
            grid.Children.Add(image);
            grid.Children.Add(new DrawingController());
            Content = grid;
            DataContext = FakeDocument.Global;
        }
        protected abstract IRenderTarget GetRenderTarget();
        protected virtual void Update(FakeDocument document)
        {
            var vertexbuffer = m_device.CreateVertexBuffer((uint)(Marshal.SizeOf(typeof(Vertex)) * document.context.vertexCount()), (uint)(Marshal.SizeOf(typeof(Vertex))), document.context.vertexPointer());
            var indexbuffer = m_device.CreateIndexBuffer(sizeof(uint) * document.context.indexCount(), document.context.indexPointer());
            m_device.BeginRender();
            m_device.BeginPass(GetRenderTarget(), new Color());
            m_device.SetViewport(new Viewport { width = 256, height = 256, maxDepth = 1 });
            m_device.SetShader(m_shader);
            m_device.SetTexture(m_texture);
            m_device.SetVertexBuffer(vertexbuffer, (uint)Marshal.SizeOf(typeof(Vertex)));
            m_device.SetIndexBuffer(indexbuffer);
            m_device.DrawIndexedPrimitives(document.context.vertexCount(), document.context.indexCount());
            m_device.EndPass();
            m_device.EndRender();
        }
        static void DocumentChanged(object s, DependencyPropertyChangedEventArgs e)
        {
            if (!(s is DrawingViewD3DImage view))
            {
                return;
            }
            if (e.OldValue is FakeDocument old)
            {
                old.PropertyChanged -= view.Update;
            }
            if (e.NewValue is FakeDocument newdoc)
            {
                newdoc.PropertyChanged += view.Update;
            }
        }
        void Update()
        {
            var document = DataContext as FakeDocument;
            if (document == null)
            {
                return;
            }
            Update(document);
            m_d3dImage.Lock();
            m_d3dImage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, m_renderTarget9.GetIDirect3DSurface9Pointer());
            m_d3dImage.AddDirtyRect(new Int32Rect(0, 0, m_d3dImage.PixelWidth, m_d3dImage.PixelHeight));
            m_d3dImage.Unlock();
        }
        void Update(object s, PropertyChangedEventArgs e)
        {
            Update();
        }
        protected static RenderTargetDeclaration m_renderTargetDeclaration = new RenderTargetDeclaration { width = 256, height = 256 };
        protected IRenderTarget_D3D9 m_renderTarget9 = Direct3D9.Device.CreateRenderTarget(m_renderTargetDeclaration);
        protected D3DImage m_d3dImage = new D3DImage();
        DispatcherTimer m_timer = new DispatcherTimer();
        protected IDevice3D m_device;
        protected IShader m_shader;
        protected ITexture m_texture;
        protected IConstantBuffer m_constantBuffer;
        protected IConstantBufferView m_constantBufferView;
    }
}