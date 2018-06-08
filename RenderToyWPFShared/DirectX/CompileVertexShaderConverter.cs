////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using System;
using System.Globalization;
using System.Windows.Data;

namespace RenderToy.WPF
{
    public class CompileVertexShaderConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            string inputcode = value as string;
            if (inputcode == null) return null;
            try
            {
                return HLSLExtension.CompileHLSL(inputcode, "vs", "vs_3_0");
            }
            catch (Exception)
            {
                return null;
            }
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}