using RenderToy.PipelineModel;
using System;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    interface ITexture2D
    {
        Vector4D SampleTexture(double u, double v);
    }
    class TextureBrick : ITexture2D
    {
        double Brickness(double u, double v)
        {
            const double MortarWidth = 0.025;
            u = u - Math.Floor(u);
            v = v - Math.Floor(v);
            if (v < MortarWidth) return 0;
            else if (v < 0.5 - MortarWidth)
            {
                if (u < MortarWidth) return 0;
                else if (u < 1.0 - MortarWidth) return 1;
                else return 0;
            }
            else if (v < 0.5 + MortarWidth) return 0;
            else if (v < 1.0 - MortarWidth)
            {
                if (u < 0.5 - MortarWidth) return 1;
                else if (u < 0.5 + MortarWidth) return 0;
                else return 1;
            }
            else return 0;
        }
        public Vector4D SampleTexture(double u, double v)
        {
            // Calculate base noise.
            double n = TexturePerlin.PerlinNoise2D(u * 16, v * 16);
            n = MathHelp.Clamp(n, -1, 1) * 0.1;
            double m = TexturePerlin.PerlinNoise2D(u * 64, v * 512);
            m = MathHelp.Clamp(m, -1, 1) * 0.2;
            Vector4D BrickColor = new Vector4D(0.5 + m + n, 0.0, 0.0, 1);
            m = TexturePerlin.PerlinNoise2D(u * 512, v * 512);
            m = MathHelp.Clamp(m, -1, 1) * 0.1;
            Vector4D MortarColor = new Vector4D(0.4 + m + n, 0.4 + m + n, 0.4 + m + n, 1);
            double bn = TexturePerlin.PerlinNoise2D(u * 64, v * 64);
            bn = MathHelp.Clamp(bn, 0, 1) * 1.25;
            // Calculate brickness mask.
            double brickness = Brickness(u, v) - bn;
            return brickness < 0.5 ? MortarColor : BrickColor;
        }
    }
    class TexturePerlin : ITexture2D
    {
        static double Interpolate(double a, double b, double x)
        {
            return a * (1 - x) + b * x;
        }
        static double Noise1(int x, int y)
        {
            int n = x + y * 57;
            n = (n<<13) ^ n;
            return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
        }
        static double Noise(double x, double y)
        {
            return Noise1((int)x, (int)y);
        }
        static double SmoothNoise(double x, double y)
        {
            double corners = (Noise(x - 1, y - 1) + Noise(x + 1, y - 1) + Noise(x - 1, y + 1) + Noise(x + 1, y + 1)) / 16;
            double sides = (Noise(x - 1, y) + Noise(x + 1, y) + Noise(x, y - 1) + Noise(x, y + 1)) / 8;
            double center = Noise(x, y) / 4;
            return corners + sides + center;
        }
        static double InterpolatedNoise(double x, double y)
        {
            int ix = (int)x;
            double fx = x - ix;
            int iy = (int)y;
            double fy = y - iy;
            double v1 = SmoothNoise(ix, iy);
            double v2 = SmoothNoise(ix + 1, iy);
            double v3 = SmoothNoise(ix, iy + 1);
            double v4 = SmoothNoise(ix + 1, iy + 1);
            double i1 = Interpolate(v1, v2, fx);
            double i2 = Interpolate(v3, v4, fx);
            return Interpolate(i1, i2, fy);
        }
        public static double PerlinNoise2D(double x, double y)
        {
            double sum = 0;
            for (int i = 0; i < 4; ++i)
            {
                double frequency = Math.Pow(2, i);
                double amplitude = Math.Pow(0.5, i);
                sum = sum + InterpolatedNoise(x * frequency, y * frequency) * amplitude;
            }
            return sum;
        }
        public Vector4D SampleTexture(double u, double v)
        {
            double p = PerlinNoise2D(u * 256, v * 256);
            return new Vector4D(p, p, p, 1);
        }
    }
    class ViewTexture : FrameworkElement
    {
        public ViewTexture()
        {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
            ClipToBounds = true;
            Texture = new TextureBrick();
        }
        ITexture2D Texture
        {
            set
            {
                texture = value;
                InvalidateVisual();
                bitmap = null;
                if (texture == null) return;
                bitmap = new WriteableBitmap(bitmapWidth, bitmapHeight, 0, 0, PixelFormats.Bgra32, null);
                bitmap.Lock();
                unsafe
                {
                    for (int y = 0; y < bitmapHeight; ++y)
                    {
                        void* raster = (byte*)bitmap.BackBuffer.ToPointer() + bitmap.BackBufferStride * y;
                        for (int x = 0; x < bitmapWidth; ++x)
                        {
                            ((uint*)raster)[x] = Rasterization.ColorToUInt32(texture.SampleTexture((x + 0.5) * 8 / bitmapWidth, (y + 0.5) * 8 / bitmapHeight));
                        }
                    }
                }
                bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmapWidth, bitmapHeight));
                bitmap.Unlock();
            }
        }
        Transform Transform
        {
            get
            {
                TransformGroup transform = new TransformGroup();
                transform.Children.Add(new TranslateTransform(-bitmapWidth / 2, -bitmapHeight / 2));
                transform.Children.Add(new TranslateTransform(translatex, translatey));
                transform.Children.Add(new ScaleTransform(scale, scale));
                transform.Children.Add(new TranslateTransform(ActualWidth / 2, ActualHeight / 2));
                return transform;
            }
        }
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            if (dragActive) return;
            Focus();
            CaptureMouse();
            dragActive = true;
            dragPoint = Transform.Inverse.Transform(e.GetPosition(this));
        }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            ReleaseMouseCapture();
            dragActive = false;
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!dragActive) return;
            Point mouse = e.GetPosition(this);
            translatex = (mouse.X - ActualWidth / 2) / scale - dragPoint.X + bitmapWidth / 2;
            translatey = (mouse.Y - ActualHeight / 2) / scale - dragPoint.Y + bitmapHeight / 2;
            InvalidateVisual();
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            if (e.Delta < 0)
            {
                scale /= 2;
            }
            if (e.Delta > 0)
            {
                scale *= 2;
            }
            InvalidateVisual();
            if (!dragActive) return;
            Point mouse = e.GetPosition(this);
            translatex = (mouse.X - ActualWidth / 2) / scale - dragPoint.X + bitmapWidth / 2;
            translatey = (mouse.Y - ActualHeight / 2) / scale - dragPoint.Y + bitmapHeight / 2;
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            if (bitmap == null) return;
            drawingContext.PushTransform(Transform);
            drawingContext.DrawImage(bitmap, new Rect(0, 0, bitmapWidth, bitmapHeight));
            drawingContext.Pop();
        }
        ITexture2D texture = null;
        WriteableBitmap bitmap = null;
        const int bitmapWidth = 1024;
        const int bitmapHeight = 1024;
        double translatex = 0;
        double translatey = 0;
        double scale = 1;
        bool dragActive = false;
        Point dragPoint;
    }
}