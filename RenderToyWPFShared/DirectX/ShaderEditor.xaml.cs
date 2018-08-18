using RenderToy.Expressions;
using RenderToy.Shaders;
using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Threading;

namespace RenderToy.WPF
{
    public sealed partial class ShaderEditor : UserControl
    {
        public static readonly DependencyProperty ProfileVSProperty = DependencyProperty.Register("ProfileVS", typeof(string), typeof(ShaderEditor));
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
        public static readonly DependencyProperty ProfilePSProperty = DependencyProperty.Register("ProfilePS", typeof(string), typeof(ShaderEditor));
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
            Action Compile = async () =>
            {
                var code = CodeEditor.Text;
                var profilevs = ProfileVS;
                var profileps = ProfilePS;
                string errors = "";
                try
                {
                    var bytecode = await HLSLExtensions.CompileHLSLAsync(code, "vs", profilevs);
                    Dispatcher.Invoke(() => SetValue(BytecodeVSProperty, bytecode));
                }
                catch (Exception exception)
                {
                    Dispatcher.Invoke(() => SetValue(BytecodeVSProperty, null));
                    errors = errors + exception.Message;
                }
                try
                {
                    var bytecode = await HLSLExtensions.CompileHLSLAsync(code, "ps", profileps);
                    Dispatcher.Invoke(() => SetValue(BytecodePSProperty, bytecode));
                }
                catch (Exception exception)
                {
                    Dispatcher.Invoke(() => SetValue(BytecodePSProperty, null));
                    errors = errors + exception.Message;
                }
                var errordefinitions = ErrorDefinition.GetErrors(errors).Distinct().ToArray();
                Dispatcher.Invoke(() => adornertextboxfloaters.SetErrors(errordefinitions));
            };
            CodeEditor.TextChanged += (s, e) =>
            {
                recompiletimer.Stop();
                recompiletimer.Start();
            };
            Compile();
            recompiletimer = new DispatcherTimer(TimeSpan.FromMilliseconds(250), DispatcherPriority.ApplicationIdle, (s, e) => { recompiletimer.Stop(); Compile(); }, Dispatcher);
        }
        DispatcherTimer recompiletimer;
    }
}
