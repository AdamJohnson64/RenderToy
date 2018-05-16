using RenderToy.Materials;
using RenderToy.Utility;
using System;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Media;

namespace RenderToy.WPF
{
    interface INodeInputHandle
    {
        Point GetInputHandleLocation(PropertyInfo p);
    }
    class ViewMaterialNode : ContentPresenter, INodeInputHandle
    {
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            var penblack = new Pen(Brushes.Black, 1);
            var typeface = new Typeface("Arial");
            if (Content == null) return;
            var properties =
                Content.GetType().GetProperties()
                .Where(i => typeof(IMaterial).IsAssignableFrom(i.PropertyType))
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
            if (Content == null) return new Size(0, 0);
            var interior = GetVisualChild(0) as UIElement;
            if (interior == null) return new Size(0, 0);
            var properties =
                Content.GetType().GetProperties()
                .Where(i => typeof(IMaterial).IsAssignableFrom(i.PropertyType));
            if (properties.Count() == 0)
            {
                // If we have no inputs then don't reserve 64 pixels for the input interfaces.
                interior.Measure(new Size(constraint.Width, constraint.Height));
                return new Size(interior.DesiredSize.Width, interior.DesiredSize.Height);
            }
            else
            {
                // Reserve 64 pixels to draw the input interfaces.
                interior.Measure(new Size(constraint.Width - 64, constraint.Height));
                return new Size(interior.DesiredSize.Width + 64, interior.DesiredSize.Height);
            }
        }
        public Point GetInputHandleLocation(PropertyInfo p)
        {
            if (Content == null) return new Point(0, 0);
            var properties =
                Content.GetType().GetProperties()
                .Where(i => typeof(IMaterial).IsAssignableFrom(i.PropertyType))
                .Select((i, v) => new { Property = i, Index = v })
                .ToArray();
            var find = properties
                .FirstOrDefault(i => i.Property == p);
            if (find == null) return new Point(0, 0);
            return new Point(ActualWidth - 4, (find.Index + 0.5) * ActualHeight / properties.Length);
        }
    }
    class NamedConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null) return "NULL";
            var named = value as INamed;
            if (named != null) return named.GetName();
            return value.GetType().Name;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}