using RenderToy.MaterialNetwork;
using RenderToy.PipelineModel;
using RenderToy.Textures;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewIMNodeThumbnail : FrameworkElement
    {
        #region - Section : Dependency Properties -
        static DependencyProperty ThumbnailSourceProperty = DependencyProperty.Register("ThumbnailSource", typeof(IMNNode), typeof(ViewIMNodeThumbnail), new FrameworkPropertyMetadata(null, InvalidateBitmap));
        public IMNNode ThumbnailSource
        {
            get { return (IMNNode)GetValue(ThumbnailSourceProperty); }
            set { SetValue(ThumbnailSourceProperty, value); }
        }
        static DependencyProperty ThumbnailWidthProperty = DependencyProperty.Register("ThumbnailWidth", typeof(int), typeof(ViewIMNodeThumbnail), new FrameworkPropertyMetadata(32, FrameworkPropertyMetadataOptions.AffectsMeasure, InvalidateBitmap));
        public int ThumbnailWidth
        {
            get { return (int)GetValue(ThumbnailWidthProperty); }
            set { SetValue(ThumbnailWidthProperty, value); }
        }
        static DependencyProperty ThumbnailHeightProperty = DependencyProperty.Register("ThumbnailHeight", typeof(int), typeof(ViewIMNodeThumbnail), new FrameworkPropertyMetadata(32, FrameworkPropertyMetadataOptions.AffectsMeasure, InvalidateBitmap));
        public int ThumbnailHeight
        {
            get { return (int)GetValue(ThumbnailHeightProperty); }
            set { SetValue(ThumbnailHeightProperty, value); }
        }
        static void InvalidateBitmap(object s, DependencyPropertyChangedEventArgs e)
        {
            ((ViewIMNodeThumbnail)s).InvalidateBitmap();
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
            if (bitmap == null) return;
            drawingContext.DrawImage(bitmap, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        #endregion
    }
    class ViewMaterialNetwork : FrameworkElement
    {
        public ViewMaterialNetwork()
        {
            ClipToBounds = true;
            Network = TextureBrick.Create();
        }
        IMNNode<Vector4D> Network
        {
            set
            {
                network = value;
                InvalidateVisual();
            }
        }
        IMNNode<Vector4D> network;
        protected override void OnRender(DrawingContext drawingContext)
        {
            const int bitmapWidth = 64;
            const int bitmapHeight = 64;
            var typeface = new Typeface("Arial");
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            int span = 0;
            foreach (var node in EnumerateNodes(network))
            {
                System.Type type = node.GetType();
                if (typeof(IMNNode<double>).IsAssignableFrom(type))
                {
                    var cast = (IMNNode<double>)node;
                    if (cast.IsConstant())
                    {
                        var formattedtext = new FormattedText(type.Name + " = " + cast.Eval(new EvalContext()).ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 10, Brushes.Black);
                        drawingContext.DrawText(formattedtext, new Point(0, span));
                        span += 10;
                    }
                    else
                    {
                        var formattedtext = new FormattedText(type.Name, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 10, Brushes.Black);
                        drawingContext.DrawText(formattedtext, new Point(0, span));
                        span += 10;
                        var bitmap = ViewIMNodeThumbnail.CreateIMNodeThumbnailDouble(cast, 128, 128);
                        drawingContext.DrawImage(bitmap, new Rect(0, span, bitmapWidth, bitmapHeight));
                        span += bitmapHeight;
                    }
                }
                if (typeof(IMNNode<Vector4D>).IsAssignableFrom(type))
                {
                    var cast = (IMNNode<Vector4D>)node;
                    var bitmap = ViewIMNodeThumbnail.CreateIMNodeThumbnailV4D(cast, 128, 128);
                    drawingContext.DrawImage(bitmap, new Rect(0, 0, bitmapWidth, bitmapHeight));
                    span += bitmapHeight;
                }
                var properties = type.GetProperties();
                for (int i = 0; i < properties.Length; ++i)
                {
                    var text = properties[i].Name + " = " + properties[i].GetValue(node);
                    var formattedtext = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 10, Brushes.Black);
                    drawingContext.DrawText(formattedtext, new Point(0, span));
                    span += 10;
                }
            }
        }
        public static IEnumerable<IMNNode> EnumerateNodes(IMNNode node)
        {
            yield return node;
            System.Type type = node.GetType();
            foreach (var property in type.GetProperties())
            {
                if (typeof(IMNNode).IsAssignableFrom(property.PropertyType))
                {
                    foreach (var next in EnumerateNodes((IMNNode)property.GetValue(node)))
                    {
                        yield return next;
                    }
                }
            }
        }
    }
}
