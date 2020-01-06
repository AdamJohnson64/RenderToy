using RenderToy.WPF;
using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Threading;

namespace Arcturus.Managed
{
    public class DrawingViewReference : ViewD3DImage
    {
        static DrawingViewReference()
        {
            //DataContextProperty.OverrideMetadata(typeof(DrawingViewReference), new FrameworkPropertyMetadata(DocumentChanged));
        }
        public DrawingViewReference()
        {
            DataContext = FakeDocument.Global;
            m_timer.Interval = TimeSpan.FromSeconds(1.0);
            m_timer.Tick += (s, e) => UpdateThread();
            m_timer.Start();
        }
        void Update(FakeDocument document)
        {
            int width = Target.PixelWidth;
            int height = Target.PixelHeight;
            DrawingContextReference dc = new DrawingContextReference();
            document.Execute(dc);
            uint[] pixels = new uint[width * height];
            unsafe
            {
                fixed (uint* p = &pixels[0])
                {
                    dc.renderTo(new IntPtr(p), (uint)width, (uint)height, 4 * (uint)width);
                    d3d9backbuffer.Fill(new IntPtr(p));
                }
            }
            Target.Lock();
            Target.SetBackBuffer(D3DResourceType.IDirect3DSurface9, d3d9backbuffer.GetIDirect3DSurface9Pointer());
            Target.AddDirtyRect(new Int32Rect(0, 0, width, height));
            Target.Unlock();
        }
        void UpdateThread()
        {
            if (!IsVisible)
            {
                return;
            }
            var document = DataContext as FakeDocument;
            if (document == null)
            {
                return;
            }
            Update(document);
        }
        void UpdateThread(object s, PropertyChangedEventArgs e)
        {
            UpdateThread();
        }
        static void DocumentChanged(object s, DependencyPropertyChangedEventArgs e)
        {
            if (!(s is DrawingViewReference view))
            {
                return;
            }
            if (e.OldValue is FakeDocument old)
            {
                old.PropertyChanged -= view.UpdateThread;
            }
            if (e.NewValue is FakeDocument newdoc)
            {
                newdoc.PropertyChanged += view.UpdateThread;
            }
        }
        DispatcherTimer m_timer = new DispatcherTimer();
    }
}