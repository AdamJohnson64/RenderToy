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
    class ViewMaterial : FrameworkElement
    {
        #region - Section : Dependency Properties -
        static DependencyProperty MaterialSourceProperty = DependencyProperty.Register("MaterialSource", typeof(IMNNode), typeof(ViewMaterial), new FrameworkPropertyMetadata(null, InvalidateBitmap));
        public IMNNode MaterialSource
        {
            get { return (IMNNode)GetValue(MaterialSourceProperty); }
            set { SetValue(MaterialSourceProperty, value); }
        }
        static DependencyProperty MaterialWidthProperty = DependencyProperty.Register("MaterialWidth", typeof(int), typeof(ViewMaterial), new FrameworkPropertyMetadata(32, FrameworkPropertyMetadataOptions.AffectsMeasure, InvalidateBitmap));
        public int MaterialWidth
        {
            get { return (int)GetValue(MaterialWidthProperty); }
            set { SetValue(MaterialWidthProperty, value); }
        }
        static DependencyProperty MaterialHeightProperty = DependencyProperty.Register("MaterialHeight", typeof(int), typeof(ViewMaterial), new FrameworkPropertyMetadata(32, FrameworkPropertyMetadataOptions.AffectsMeasure, InvalidateBitmap));
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
        void InvalidateBitmap()
        {
            bitmap = MaterialBitmapConverter.ConvertToBitmap(MaterialSource, MaterialWidth, MaterialHeight);
            InvalidateVisual();
        }
        WriteableBitmap bitmap = null;
        #endregion
        #region - Section : UIElement Overrides -
        protected override Size MeasureOverride(Size availableSize)
        {
            return new Size(Math.Min(availableSize.Width, MaterialWidth), Math.Min(availableSize.Height, MaterialHeight));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            if (MaterialSource is MNTexCoordU)
            {
                var formattedtext = new FormattedText("U", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Black);
                drawingContext.DrawText(formattedtext, new Point((ActualWidth - formattedtext.Width) / 2, (ActualHeight - formattedtext.Height) / 2));
            }
            else if (MaterialSource is MNTexCoordV)
            {
                var formattedtext = new FormattedText("V", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.Black);
                drawingContext.DrawText(formattedtext, new Point((ActualWidth - formattedtext.Width) / 2, (ActualHeight - formattedtext.Height) / 2));
            }
            else if (MaterialSource is IMNNode<double> && MaterialSource.IsConstant())
            {
                EvalContext context = new EvalContext();
                double value = ((IMNNode<double>)MaterialSource).Eval(context);
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
