using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Threading;

namespace Arcturus.Managed
{
    abstract class DrawingViewD3DImage : ContentControl
    {
        static DrawingViewD3DImage()
        {
            DataContextProperty.OverrideMetadata(typeof(DrawingViewD3DImage), new FrameworkPropertyMetadata(DocumentChanged));
        }
        public DrawingViewD3DImage()
        {
            var grid = new Grid();
            var image = new Image { Source = m_d3dImage, HorizontalAlignment = HorizontalAlignment.Left, VerticalAlignment = VerticalAlignment.Top, Width = 256, Height = 256 };
            m_timer.Interval = TimeSpan.FromSeconds(1);
            m_timer.Tick += (s, e) => Update();
            m_timer.Start();
            grid.Children.Add(image);
            grid.Children.Add(new DrawingController());
            Content = grid;
        }
        protected abstract void Update(FakeDocument document);
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
    }
}