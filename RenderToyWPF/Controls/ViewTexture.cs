using RenderToy.PipelineModel;
using System;
using System.Windows;
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
        public Vector4D SampleTexture(double u, double v)
        {
            u = u - Math.Floor(u);
            v = v - Math.Floor(v);
            double m = TexturePerlin.PerlinNoise2D(u * 128, v * 128);
            m = MathHelp.Clamp(m, -1, 1) * 0.2;
            Vector4D BrickColor = new Vector4D(0.5 + m, 0.0, 0.0, 1);
            m = TexturePerlin.PerlinNoise2D(u * 512, v * 512);
            m = MathHelp.Clamp(m, -1, 1) * 0.2;
            Vector4D MortarColor = new Vector4D(0.4 + m, 0.4 + m, 0.4 + m, 1);
            const double MortarWidth = 0.025;
            if (v < MortarWidth) return MortarColor;
            else if (v < 0.5 - MortarWidth)
            {
                if (u < MortarWidth) return MortarColor;
                else if (u < 1.0 - MortarWidth) return BrickColor;
                else return MortarColor;
            }
            else if (v < 0.5 + MortarWidth) return MortarColor;
            else if (v < 1.0 - MortarWidth)
            {
                if (u < 0.5 - MortarWidth) return BrickColor;
                else if (u < 0.5 + MortarWidth) return MortarColor;
                else return BrickColor;
            }
            else return MortarColor;
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
        protected override void OnRender(DrawingContext drawingContext)
        {
            const int bitmapWidth = 512;
            const int bitmapHeight = 512;
            ITexture2D texture = new TextureBrick();
            WriteableBitmap bitmap = new WriteableBitmap(bitmapWidth, bitmapHeight, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            unsafe
            {
                for (int y = 0; y < bitmapHeight; ++y)
                {
                    void* raster = (byte*)bitmap.BackBuffer.ToPointer() + bitmap.BackBufferStride * y;
                    for (int x = 0; x < bitmapWidth; ++x)
                    {
                        ((uint*)raster)[x] = Rasterization.ColorToUInt32(texture.SampleTexture((x + 0.5) * 3 / bitmapWidth - 1, (y + 0.5) * 3 / bitmapHeight - 1));
                    }
                }
            }
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmapWidth, bitmapHeight));
            bitmap.Unlock();
            drawingContext.DrawImage(bitmap, new Rect(0, 0, bitmapWidth, bitmapHeight));
        }
    }
}