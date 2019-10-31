using RenderToy.Expressions;
using RenderToy.Materials;
using System;
using System.Globalization;
using System.Windows.Data;

namespace RenderToy.WPF
{
    class MaterialToHLSLConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var material = value as IMNNode;
            if (material == null) return "<NOT A MATERIAL>";
            var expression = MSILExtensions.GenerateMSIL(material);
            return HLSLGenerator.Emit(expression);
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}