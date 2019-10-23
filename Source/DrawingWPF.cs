using System;
using System.Windows;
using System.Windows.Controls;

namespace Arcturus.Managed
{
    class DrawingApp
    {
        [STAThread]
        public static void Main()
        {
            var wrappanel = new WrapPanel { Orientation = Orientation.Horizontal };
            wrappanel.Children.Add(new GroupBox { Content = new DrawingViewD3D12(), Header = "D3D12" });
            wrappanel.Children.Add(new GroupBox { Content = new DrawingViewVulkan(), Header = "Vulkan" });
            wrappanel.Children.Add(new GroupBox { Content = new DrawingViewDXR12(), Header = "DXR12" });
            new Application().Run(new Window { Content = wrappanel, DataContext = new FakeDocument(), Title = "Drawing (WPF)", Width = 640, Height = 640 });
        }
    }
}