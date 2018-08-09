////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Shaders;
using System.Collections.Generic;
using System.Windows;

namespace RenderToy.WPF
{
    public class ViewD3D9 : ViewD3DImageDirect
    {
        public static DependencyProperty VertexShaderProperty = DependencyProperty.Register("VertexShader", typeof(byte[]), typeof(ViewD3D9), new FrameworkPropertyMetadata(HLSL.D3D9VS, FrameworkPropertyMetadataOptions.AffectsRender));
        public byte[] VertexShader
        {
            get { return (byte[])GetValue(VertexShaderProperty); }
            set { SetValue(VertexShaderProperty, value); }
        }
        public static DependencyProperty PixelShaderProperty = DependencyProperty.Register("PixelShader", typeof(byte[]), typeof(ViewD3D9), new FrameworkPropertyMetadata(HLSL.D3D9PS, FrameworkPropertyMetadataOptions.AffectsRender));
        public byte[] PixelShader
        {
            get { return (byte[])GetValue(PixelShaderProperty); }
            set { SetValue(PixelShaderProperty, value); }
        }
        protected override void RenderD3D()
        {
            if (VertexShader == null || PixelShader == null) return;
            var vertexshader = Direct3D9Helper.device.CreateVertexShader(VertexShader);
            var pixelshader = Direct3D9Helper.device.CreatePixelShader(PixelShader);
            Direct3D9Helper.device.SetVertexShader(vertexshader);
            Direct3D9Helper.device.SetPixelShader(pixelshader);
            var constants = new Dictionary<string, object>();
            constants["transformAspect"] = Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            constants["transformCamera"] = AttachedView.GetTransformCamera(this);
            constants["transformView"] = AttachedView.GetTransformView(this);
            constants["transformProjection"] = AttachedView.GetTransformProjection(this);
            Direct3D9Helper.CreateSceneDraw(AttachedView.GetScene(this))(constants);
        }
    }
}