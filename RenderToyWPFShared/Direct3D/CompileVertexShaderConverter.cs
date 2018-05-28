////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Windows.Data;

namespace RenderToy.WPF
{
    public class CompileVertexShaderConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            string inputcode = value as string;
            if (inputcode == null) return null;
            D3DBlob code = new D3DBlob();
            Direct3DCompiler.D3DCompile(inputcode, "temp.vs", "vs", "vs_3_0", 0, 0, code, null);
            var buffer = code.GetBufferPointer();
            if (buffer == IntPtr.Zero) return null;
            var buffersize = code.GetBufferSize();
            byte[] codebytes = new byte[buffersize];
            Marshal.Copy(buffer, codebytes, 0, (int)buffersize);
            return codebytes;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}