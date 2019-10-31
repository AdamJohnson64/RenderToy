using RenderToy.Expressions;
using RenderToy.Materials;
using System;
using System.Globalization;
using System.Linq.Expressions;
using System.Reflection;
using System.Windows.Data;

namespace RenderToy.WPF
{
    class MaterialToMSILConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var material = value as IMNNode;
            if (material == null) return "<NOT A MATERIAL>";
            var expression = MSILExtensions.GenerateMSIL(material);
            var propertyInfo = typeof(Expression).GetProperty("DebugView", BindingFlags.Instance | BindingFlags.NonPublic);
            return propertyInfo.GetValue(expression) as string;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}