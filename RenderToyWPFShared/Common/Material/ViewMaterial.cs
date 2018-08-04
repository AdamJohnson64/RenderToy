////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.DirectX;
using RenderToy.Expressions;
using RenderToy.Materials;
using System.Globalization;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewMaterial : FrameworkElement
    {
        #region - Section : Dependency Properties -
        public static DependencyProperty MaterialSourceProperty = DependencyProperty.Register("MaterialSource", typeof(IMaterial), typeof(ViewMaterial), new FrameworkPropertyMetadata(null, InvalidateBitmap));
        public IMaterial MaterialSource
        {
            get { return (IMaterial)GetValue(MaterialSourceProperty); }
            set { SetValue(MaterialSourceProperty, value); }
        }
        public static DependencyProperty MaterialWidthProperty = DependencyProperty.Register("MaterialWidth", typeof(int), typeof(ViewMaterial), new FrameworkPropertyMetadata(DirectXHelper.ThumbnailSize, FrameworkPropertyMetadataOptions.AffectsMeasure, InvalidateBitmap));
        public int MaterialWidth
        {
            get { return (int)GetValue(MaterialWidthProperty); }
            set { SetValue(MaterialWidthProperty, value); }
        }
        public static DependencyProperty MaterialHeightProperty = DependencyProperty.Register("MaterialHeight", typeof(int), typeof(ViewMaterial), new FrameworkPropertyMetadata(DirectXHelper.ThumbnailSize, FrameworkPropertyMetadataOptions.AffectsMeasure, InvalidateBitmap));
        public int MaterialHeight
        {
            get { return (int)GetValue(MaterialHeightProperty); }
            set { SetValue(MaterialHeightProperty, value); }
        }
        static void InvalidateBitmap(object s, DependencyPropertyChangedEventArgs e)
        {
            ((ViewMaterial)s).InvalidateBitmap();
        }
        #endregion
        #region - Section : Image Handling -
        async void InvalidateBitmap()
        {
            bitmap = null;
            InvalidateVisual();
            var imageconverter = DirectXHelper.GetImageConverter(MaterialSource, MaterialWidth, MaterialHeight);
            var newbitmap = new WriteableBitmap(imageconverter.GetImageWidth(), imageconverter.GetImageHeight(), 0, 0, PixelFormats.Bgra32, null);
            newbitmap.Lock();
            var material = MaterialSource;
            var bitmapptr = newbitmap.BackBuffer;
            var bitmapwidth = newbitmap.PixelWidth;
            var bitmapheight = newbitmap.PixelHeight;
            var bitmapstride = newbitmap.BackBufferStride;
            await Task.Factory.StartNew(() => DirectXHelper.ConvertToBitmap(imageconverter, bitmapptr, bitmapwidth, bitmapheight, bitmapstride));
            newbitmap.AddDirtyRect(new Int32Rect(0, 0, bitmapwidth, bitmapheight));
            newbitmap.Unlock();
            bitmap = newbitmap;
            InvalidateVisual();
        }
        WriteableBitmap bitmap = null;
        #endregion
        #region - Section : UIElement Overrides -
        protected override Size MeasureOverride(Size availableSize)
        {
            return new Size(System.Math.Min(availableSize.Width, MaterialWidth), System.Math.Min(availableSize.Height, MaterialHeight));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            if (MaterialSource is MNTexCoordU)
            {
                var formattedtext = new FormattedText("U", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Black, 1.0);
                drawingContext.DrawText(formattedtext, new Point((ActualWidth - formattedtext.Width) / 2, (ActualHeight - formattedtext.Height) / 2));
            }
            else if (MaterialSource is MNTexCoordV)
            {
                var formattedtext = new FormattedText("V", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Black, 1.0);
                drawingContext.DrawText(formattedtext, new Point((ActualWidth - formattedtext.Width) / 2, (ActualHeight - formattedtext.Height) / 2));
            }
            else if (MaterialSource is IMNNode<double> && MaterialSource.IsConstant())
            {
                var lambda = ((IMNNode<double>)MaterialSource).CompileMSIL();
                double value = lambda(new EvalContext());
                var formattedtext = new FormattedText(value.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Black, 1.0);
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
