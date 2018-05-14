////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
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
            bitmap = MaterialBitmapConverter.ConvertToBitmap(ThumbnailSource, ThumbnailWidth, ThumbnailHeight);
            InvalidateVisual();
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
