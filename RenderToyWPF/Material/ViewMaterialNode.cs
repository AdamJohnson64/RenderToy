using RenderToy.SceneGraph.Materials;
using System;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Media;

namespace RenderToy.WPF
{
    class ViewMaterialNode : ContentControl
    {
        public static DependencyProperty NodeProperty = DependencyProperty.Register("Node", typeof(IMNNode), typeof(ViewMaterialNode), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender));
        public IMNNode Node
        {
            get { return (IMNNode)GetValue(NodeProperty); }
            set { SetValue(NodeProperty, value); }
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            var penblack = new Pen(Brushes.Black, 1);
            var typeface = new Typeface("Arial");
            if (Node == null) return;
            var properties =
                Node.GetType().GetProperties()
                .Where(i => typeof(IMNNode).IsAssignableFrom(i.PropertyType))
                .ToArray();
            for (int i = 0; i < properties.Length; ++i)
            {
                double y = (i + 0.5) * ActualHeight / properties.Length;
                drawingContext.DrawLine(penblack, new Point(ActualWidth - 64, y), new Point(ActualWidth, y));
                drawingContext.DrawEllipse(Brushes.White, penblack, new Point(ActualWidth - 4, y), 4, 4);
                var formattedtext = new FormattedText(properties[i].Name, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 10, Brushes.Black);
                drawingContext.DrawText(formattedtext, new Point(ActualWidth - 32 - formattedtext.Width / 2, y - formattedtext.Height));
            }
        }
        protected override Size ArrangeOverride(Size arrangeBounds)
        {
            var interior = GetVisualChild(0) as UIElement;
            if (interior == null) return arrangeBounds;
            interior.Arrange(new Rect(0, 0, interior.DesiredSize.Width, interior.DesiredSize.Height));
            return arrangeBounds;
        }
        protected override Size MeasureOverride(Size constraint)
        {
            var interior = GetVisualChild(0) as UIElement;
            if (interior == null) return new Size(0, 0);
            interior.Measure(new Size(constraint.Width - 64, constraint.Height));
            return new Size(interior.DesiredSize.Width + 64, interior.DesiredSize.Height);
        }
    }
    class ShortTypeNameConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null) return "NULL";
            return value.GetType().Name;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}