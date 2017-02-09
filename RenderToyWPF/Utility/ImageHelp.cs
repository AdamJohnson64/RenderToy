////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy
{
    public static class ImageHelp
    {
        /// <summary>
        /// Create an image from a FillFunction.
        /// </summary>
        /// <param name="fillwith">The filling function (one of the renderers).</param>
        /// <param name="scene">The scene to be drawn.</param>
        /// <param name="mvp">The MVP of the camera.</param>
        /// <param name="render_width">The desired pixel width of the output.</param>
        /// <param name="render_height">The desired pixel height of the output.</param>
        /// <returns></returns>
        public static ImageSource CreateImage(RenderCall.FillFunction fillwith, Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            WriteableBitmap bitmap = new WriteableBitmap(render_width, render_height, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            fillwith(scene, mvp, bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, render_width, render_height));
            bitmap.Unlock();
            return bitmap;
        }
    }
}