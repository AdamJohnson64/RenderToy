using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;
using System.Runtime.InteropServices.WindowsRuntime;

// The User Control item template is documented at http://go.microsoft.com/fwlink/?LinkId=234236

namespace RenderToy
{
    public sealed partial class RenderViewport : UserControl
    {
        public RenderViewport()
        {
            this.InitializeComponent();
            var bitmap = new WriteableBitmap(256, 256);
            var buffer = bitmap.PixelBuffer;
            var length = buffer.Length;
            using (var stream = bitmap.PixelBuffer.AsStream())
            {
                for (int y = 0; y < 256; ++y)
                {
                    for (int x = 0; x < 256; ++x)
                    {
                        stream.Position = 4 * x + 4 * 256 * y;
                        stream.Write(new byte[] { (byte)0, (byte)0, (byte)0, (byte)0 }, 0, 4);
                    }
                }
            }
            bitmap.Invalidate();
            Component_Image.Source = bitmap;
        }
    }
}
