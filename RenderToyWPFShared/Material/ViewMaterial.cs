////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewMaterial : FrameworkElement
    {
        #region - Section : Dependency Properties -
        public static DependencyProperty TextureProperty = DependencyProperty.Register("Texture", typeof(IMNNode), typeof(ViewMaterial), new FrameworkPropertyMetadata(null, OnInvalidateTexture));
        public IMNNode Texture
        {
            get { return (IMNNode)GetValue(TextureProperty); }
            set { SetValue(TextureProperty, value); }
        }
        static void OnInvalidateTexture(object s, DependencyPropertyChangedEventArgs e)
        {
            ((ViewMaterial)s).InvalidateTexture();
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
        WriteableBitmap bitmap = null;
        const int bitmapWidth = 256;
        const int bitmapHeight = 256;
        double translatex = 0;
        double translatey = 0;
        double scale = 1;
        bool dragActive = false;
        Point dragPoint;
        #endregion
        #region - Section : Construction -
        public ViewMaterial()
        {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
            ClipToBounds = true;
            //Texture = new TextureNetwork();
        }
        #endregion
        #region - Section : Bitmap Handling -
        void InvalidateTexture()
        {
            bitmap = MaterialBitmapConverter.ConvertToBitmap(Texture, bitmapWidth, bitmapHeight);
            InvalidateVisual();
        }
        #endregion
        #region - Section : UIElement Overrides -
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
            for (int y = -1; y <= 1; ++y)
            {
                for (int x = -1; x <= 1; ++x)
                {
                    drawingContext.PushTransform(Transform);
                    drawingContext.DrawImage(bitmap, new Rect(x * bitmapWidth, y * bitmapHeight, bitmapWidth, bitmapHeight));
                    drawingContext.Pop();
                }
            }
            drawingContext.DrawRectangle(null, new Pen(Brushes.White, 1), Rect.Transform(new Rect(0, 0, bitmapWidth, bitmapHeight), Transform.Value));
        }
        #endregion
    }
}