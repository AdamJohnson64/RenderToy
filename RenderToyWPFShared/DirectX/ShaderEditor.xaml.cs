using RenderToy.Expressions;
using RenderToy.Shaders;
using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;

namespace RenderToy.WPF
{
    public sealed partial class ShaderEditor : UserControl
    {
        public static readonly DependencyProperty ProfileVSProperty = DependencyProperty.Register("ProfileVS", typeof(string), typeof(ShaderEditor), new FrameworkPropertyMetadata("vs_3_0", FrameworkPropertyMetadataOptions.AffectsRender));
        public string ProfileVS
        {
            get { return (string)GetValue(ProfileVSProperty); }
            set { SetValue(ProfileVSProperty, value); }
        }
        public static readonly DependencyProperty BytecodeVSProperty = DependencyProperty.Register("BytecodeVS", typeof(byte[]), typeof(ShaderEditor));
        public byte[] BytecodeVS
        {
            get { return (byte[])GetValue(BytecodeVSProperty); }
        }
        public static readonly DependencyProperty ProfilePSProperty = DependencyProperty.Register("ProfilePS", typeof(string), typeof(ShaderEditor), new FrameworkPropertyMetadata("ps_3_0", FrameworkPropertyMetadataOptions.AffectsRender));
        public string ProfilePS
        {
            get { return (string)GetValue(ProfilePSProperty); }
            set { SetValue(ProfilePSProperty, value); }
        }
        public static readonly DependencyProperty BytecodePSProperty = DependencyProperty.Register("BytecodePS", typeof(byte[]), typeof(ShaderEditor));
        public byte[] BytecodePS
        {
            get { return (byte[])GetValue(BytecodePSProperty); }
        }
        public string Text
        {
            get { return CodeEditor.Text; }
            set { CodeEditor.Text = value; }
        }
        public ShaderEditor()
        {
            this.InitializeComponent();
            var adornerlayer = AdornerLayer.GetAdornerLayer(CodeEditor);
            var adornertextboxfloaters = new AdornerTextBoxErrors(CodeEditor);
            adornerlayer.Add(adornertextboxfloaters);
            CodeEditor.Text = HLSL.D3D9Standard;
            Action Compile = () =>
            {
                string errors = "";
                try
                {
                    SetValue(BytecodeVSProperty, HLSLExtensions.CompileHLSL(CodeEditor.Text, "vs", ProfileVS));
                }
                catch (Exception exception)
                {
                    SetValue(BytecodeVSProperty, null);
                    errors = errors + exception.Message;
                }
                try
                {
                    SetValue(BytecodePSProperty, HLSLExtensions.CompileHLSL(CodeEditor.Text, "ps", ProfilePS));
                }
                catch (Exception exception)
                {
                    SetValue(BytecodePSProperty, null);
                    errors = errors + exception.Message;
                }
                var errordefinitions = ErrorDefinition.GetErrors(errors).Distinct().ToArray();
                adornertextboxfloaters.SetErrors(errordefinitions);
            };
            CodeEditor.TextChanged += (s, e) =>
            {
                Compile();
            };
            Compile();
        }
    }
}
