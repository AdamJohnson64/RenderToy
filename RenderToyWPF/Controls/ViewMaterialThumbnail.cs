////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.PipelineModel;
using RenderToy.SceneGraph.Materials;
using System;
using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewMaterialThumbnail : FrameworkElement
    {
        #region - Section : Dependency Properties -
        static DependencyProperty ThumbnailSourceProperty = DependencyProperty.Register("ThumbnailSource", typeof(IMNNode), typeof(ViewMaterialThumbnail), new FrameworkPropertyMetadata(null, InvalidateBitmap));
        public IMNNode ThumbnailSource
        {
            get { return (IMNNode)GetValue(ThumbnailSourceProperty); }
            set { SetValue(ThumbnailSourceProperty, value); }
        }
        static DependencyProperty ThumbnailWidthProperty = DependencyProperty.Register("ThumbnailWidth", typeof(int), typeof(ViewMaterialThumbnail), new FrameworkPropertyMetadata(32, FrameworkPropertyMetadataOptions.AffectsMeasure, InvalidateBitmap));
        public int ThumbnailWidth
        {
            get { return (int)GetValue(ThumbnailWidthProperty); }
            set { SetValue(ThumbnailWidthProperty, value); }
        }
        static DependencyProperty ThumbnailHeightProperty = DependencyProperty.Register("ThumbnailHeight", typeof(int), typeof(ViewMaterialThumbnail), new FrameworkPropertyMetadata(32, FrameworkPropertyMetadataOptions.AffectsMeasure, InvalidateBitmap));
        public int ThumbnailHeight
        {
            get { return (int)GetValue(ThumbnailHeightProperty); }
            set { SetValue(ThumbnailHeightProperty, value); }
        }
        static void InvalidateBitmap(object s, DependencyPropertyChangedEventArgs e)
        {
            ((ViewMaterialThumbnail)s).InvalidateBitmap();
        }
        #endregion
        #region - Section : Thumbnail Handling -
        void InvalidateBitmap()
        {
            bitmap = CreateIMNodeThumbnail(ThumbnailSource, ThumbnailWidth, ThumbnailHeight);
            InvalidateVisual();
        }
        public static WriteableBitmap CreateIMNodeThumbnail(IMNNode node, int bitmapWidth, int bitmapHeight)
        {
            if (node == null) return null;
            System.Type type = node.GetType();
            if (typeof(IMNNode<double>).IsAssignableFrom(type))
            {
                return CreateIMNodeThumbnailDouble((IMNNode<double>)node, bitmapWidth, bitmapHeight);
            }
            else if (typeof(IMNNode<Vector4D>).IsAssignableFrom(type))
            {
                return CreateIMNodeThumbnailV4D((IMNNode<Vector4D>)node, bitmapWidth, bitmapHeight);
            }
            return null;
        }
        public static WriteableBitmap CreateIMNodeThumbnailDouble(IMNNode<double> node, int bitmapWidth, int bitmapHeight)
        {
            if (node == null || bitmapWidth == 0 || bitmapHeight == null) return null;
            var bitmap = new WriteableBitmap(bitmapWidth, bitmapHeight, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            var context = new EvalContext();
            unsafe
            {
                for (int y = 0; y < bitmapHeight; ++y)
                {
                    void* raster = (byte*)bitmap.BackBuffer.ToPointer() + bitmap.BackBufferStride * y;
                    for (int x = 0; x < bitmapWidth; ++x)
                    {
                        context.U = (x + 0.5) / bitmapWidth;
                        context.V = (y + 0.5) / bitmapHeight;
                        double v = node.Eval(context);
                        ((uint*)raster)[x] = Rasterization.ColorToUInt32(new Vector4D(v, v, v, 1));
                    }
                }
            }
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmapWidth, bitmapHeight));
            bitmap.Unlock();
            return bitmap;
        }
        public static WriteableBitmap CreateIMNodeThumbnailV4D(IMNNode<Vector4D> node, int bitmapWidth, int bitmapHeight)
        {
            if (node == null || bitmapWidth == 0 || bitmapHeight == null) return null;
            var bitmap = new WriteableBitmap(bitmapWidth, bitmapHeight, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            var context = new EvalContext();
            unsafe
            {
                for (int y = 0; y < bitmapHeight; ++y)
                {
                    void* raster = (byte*)bitmap.BackBuffer.ToPointer() + bitmap.BackBufferStride * y;
                    for (int x = 0; x < bitmapWidth; ++x)
                    {
                        context.U = (x + 0.5) / bitmapWidth;
                        context.V = (y + 0.5) / bitmapHeight;
                        ((uint*)raster)[x] = Rasterization.ColorToUInt32(node.Eval(context));
                    }
                }
            }
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmapWidth, bitmapHeight));
            bitmap.Unlock();
            return bitmap;
        }
        WriteableBitmap bitmap = null;
        #endregion
        #region - Section : UIElement Overrides -
        protected override Size MeasureOverride(Size availableSize)
        {
            return new Size(Math.Min(availableSize.Width, ThumbnailWidth), Math.Min(availableSize.Height, ThumbnailHeight));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            if (ThumbnailSource is MNTexCoordU)
            {
                var formattedtext = new FormattedText("U", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Black);
                drawingContext.DrawText(formattedtext, new Point((ActualWidth - formattedtext.Width) / 2, (ActualHeight - formattedtext.Height) / 2));
            }
            else if (ThumbnailSource is MNTexCoordV)
            {
                var formattedtext = new FormattedText("V", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Black);
                drawingContext.DrawText(formattedtext, new Point((ActualWidth - formattedtext.Width) / 2, (ActualHeight - formattedtext.Height) / 2));
            }
            else if (ThumbnailSource is IMNNode<double> && ThumbnailSource.IsConstant())
            {
                EvalContext context = new EvalContext();
                double value = ((IMNNode<double>)ThumbnailSource).Eval(context);
                var formattedtext = new FormattedText(value.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Black);
                drawingContext.DrawText(formattedtext, new Point((ActualWidth - formattedtext.Width) / 2, (ActualHeight - formattedtext.Height) / 2));
            }
            else
            {
                if (bitmap == null) return;
                drawingContext.DrawImage(bitmap, new Rect(0, 0, ActualWidth, ActualHeight));
            }
        }
        #endregion
    }
}
