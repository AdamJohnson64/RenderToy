////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using Microsoft.Win32;
using RenderToy.Materials;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Transforms;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Data;
using System.Windows.Input;

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
    public class CompilePixelShaderConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            string inputcode = value as string;
            if (inputcode == null) return null;
            D3DBlob code = new D3DBlob();
            Direct3DCompiler.D3DCompile(inputcode, "temp.ps", "ps", "ps_3_0", 0, 0, code, null);
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
    public partial class MainWindow : Window
    {
        public static RoutedUICommand CommandSceneNew = new RoutedUICommand("New Scene", "CommandSceneNew", typeof(View3DUser));
        public static RoutedUICommand CommandSceneOpen = new RoutedUICommand("Open Scene", "CommandSceneLoad", typeof(View3DUser));
        public static RoutedUICommand CommandScenePlane = new RoutedUICommand("Open Plane", "CommandScenePlane", typeof(View3DUser));
        public static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(View3DUser));
        public static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(View3DUser));
        public static RoutedUICommand CommandDebugToolPerformanceTrace = new RoutedUICommand("Performance Trace Tool (Debug)", "CommandDebugToolPerformanceTrace", typeof(View3DUser));
        public MainWindow()
        {
            InitializeComponent();
            CommandBindings.Add(new CommandBinding(CommandSceneNew, (s, e) => {
                DataContext = Document.Default;
                e.Handled = true;
            }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandSceneOpen, (s, e) => {
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.Title = "Choose Model File";
                if (ofd.ShowDialog() == true)
                {
                    var scene = new Scene();
                    if (Path.GetExtension(ofd.FileName).ToUpperInvariant() == ".BPT")
                    {
                        foreach (var primitive in LoaderBPT.LoadFromPath(ofd.FileName))
                        {
                            scene.AddChild(new Node("Bezier Patch", new TransformMatrix(Matrix3D.Identity), primitive, StockMaterials.White, StockMaterials.PlasticWhite));
                        }
                    }
                    else if (Path.GetExtension(ofd.FileName).ToUpperInvariant() == ".OBJ")
                    {
                        foreach (var node in LoaderOBJ.LoadFromPath(ofd.FileName))
                        {
                            scene.AddChild(node);
                        }
                    }
                    else if (Path.GetExtension(ofd.FileName).ToUpperInvariant() == ".PLY")
                    {
                        scene.AddChild(new Node(Path.GetFileName(ofd.FileName), new TransformMatrix(MathHelp.CreateMatrixScale(100, 100, 100)), LoaderPLY.LoadFromPath(ofd.FileName), StockMaterials.White, StockMaterials.PlasticWhite));
                    }
                    DataContext = new Document(scene);
                }
                e.Handled = true;
            }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandScenePlane, (s, e) =>
            {
                var scene = new Scene();
                var mesh = new Mesh();
                mesh.Vertices = new MeshChannel<Vector3D>(new Vector3D[] { new Vector3D(-10, 0, 10), new Vector3D(10, 0, 10), new Vector3D(10, 0, -10), new Vector3D(-10, 0, -10) }, new int[] { 0, 1, 2, 0, 2, 3 });
                mesh.Normals = new MeshChannel<Vector3D>(new Vector3D[] { new Vector3D(0, 1, 0) }, new int[] { 0, 0, 0, 0, 0, 0 });
                mesh.TexCoords = new MeshChannel<Vector2D>(new Vector2D[] { new Vector2D(0, 0), new Vector2D(4, 0), new Vector2D(4, 4), new Vector2D(0, 4) }, new int[] { 0, 1, 2, 0, 2, 3 });
                mesh.GenerateTangentSpace();
                var material = new LoaderOBJ.OBJMaterial();
                material.map_Kd = StockMaterials.Brick;
                var displace = new MNSubtract {
                    Lhs = new BrickMask { U = new MNMultiply { Lhs = new MNTexCoordU(), Rhs = new MNConstant { Value = 4 } }, V = new MNMultiply { Lhs = new MNTexCoordV(), Rhs = new MNConstant { Value = 4 } } },
                    Rhs = new MNMultiply { Lhs = new Perlin2D { U = new MNMultiply { Lhs = new MNTexCoordU(), Rhs = new MNConstant { Value = 512 } }, V = new MNMultiply { Lhs = new MNTexCoordV(), Rhs = new MNConstant { Value = 512 } } }, Rhs = new MNConstant { Value = 0.001 } }
                };
                material.map_bump = new BumpGenerate { Displacement = displace };
                material.displacement = new MNVector4D { R = displace, G = displace, B = displace, A = new MNConstant { Value = 1 } };
                scene.AddChild(new Node("Plane", new TransformMatrix(Matrix3D.Identity), mesh, StockMaterials.White, material));
                DataContext = new Document(scene);
            }));
            CommandBindings.Add(new CommandBinding(CommandRenderPreviewsToggle, (s, e) => { ViewPreview.Visibility = ViewPreview.Visibility == Visibility.Hidden ? Visibility.Visible : Visibility.Hidden; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframeToggle, (s, e) => { ViewWireframe.Visibility = ViewWireframe.Visibility == Visibility.Hidden ? Visibility.Visible : Visibility.Hidden; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandDebugToolPerformanceTrace, (s, e) => {
                var window = new Window { Title = "Performance Trace Tool", Content = new PerformanceTrace() };
                window.ShowDialog();
                e.Handled = true;
            }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            InputBindings.Add(new KeyBinding(CommandSceneNew, Key.N, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandSceneOpen, Key.O, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderPreviewsToggle, Key.P, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframeToggle, Key.W, ModifierKeys.Control));
            DataContext = Document.Default;
            ShaderCode.TextChanged += (s, e) =>
            {
                {
                    D3DBlob code = new D3DBlob();
                    D3DBlob error = new D3DBlob();
                    Direct3DCompiler.D3DCompile(ShaderCode.Text, "temp.vs", "vs", "vs_3_0", 0, 0, code, error);
                    var buffer = error.GetBufferPointer();
                    var buffersize = error.GetBufferSize();
                    ShaderErrorsVS.Text = buffer == IntPtr.Zero ? "Vertex Shader Compilation Successful." : Marshal.PtrToStringAnsi(buffer, (int)buffersize - 1);
                }
                {
                    D3DBlob code = new D3DBlob();
                    D3DBlob error = new D3DBlob();
                    Direct3DCompiler.D3DCompile(ShaderCode.Text, "temp.ps", "ps", "ps_3_0", 0, 0, code, error);
                    var buffer = error.GetBufferPointer();
                    var buffersize = error.GetBufferSize();
                    ShaderErrorsPS.Text = buffer == IntPtr.Zero ? "Pixel Shader Compilation Successful." : Marshal.PtrToStringAnsi(buffer, (int)buffersize - 1);
                }
            };
            ShaderCode.Text =
@"float4x4 ModelViewProjection : register(c0);
sampler2D SamplerAlbedo : register(s0);
sampler2D SamplerMask : register(s1);
sampler2D SamplerBump : register(s2);

struct VS_INPUT {
    float4 Position : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 Color : COLOR;
    float3 Tangent : TANGENT;
    float3 Bitangent : BINORMAL;
};

struct VS_OUTPUT {
    float4 Position : SV_Position;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 Color : COLOR;
    float3 Tangent : TANGENT;
    float3 Bitangent : BINORMAL;
};

VS_OUTPUT vs(VS_INPUT input) {
    VS_OUTPUT result;
    result.Position = mul(ModelViewProjection, input.Position);
    result.Normal = input.Normal;
    result.TexCoord = input.TexCoord;
    result.Color = input.Color;
    result.Tangent = input.Tangent;
    result.Bitangent = input.Bitangent;
    return result;
}

float4 ps(VS_OUTPUT input) : SV_Target {
    if (tex2D(SamplerMask, input.TexCoord).r < 0.5) discard;
    float3x3 tbn = {input.Tangent, input.Bitangent, input.Normal};
    float3 bump = normalize(tex2D(SamplerBump, input.TexCoord).rgb * 2 - 1);
    float3 normal = mul(bump, tbn);
    float light = clamp(dot(normal, normalize(float3(1,1,1))), 0, 1);
    return float4(light * tex2D(SamplerAlbedo, input.TexCoord).rgb, 1);
}";
        }
    }
    class Document
    {
        public IScene Scene { get; private set; }
        public ObservableCollection<IMaterial> Materials { get; private set; }
        public static Document Default = new Document(TestScenes.DefaultScene);
        public Document(IScene scene)
        {
            Scene = scene;
            var materials = EnumerateSceneRoot(scene)
                .Select(i => i.GetMaterial())
                .OfType<IMaterial>()
                .Distinct();
            //.SelectMany(i => EnumerateNodes(i));
            Materials = new ObservableCollection<IMaterial>(materials);
        }
        static IEnumerable<INode> EnumerateSceneRoot(IScene root)
        {
            if (root == null) yield break;
            foreach (var child in root.GetChildren())
            {
                foreach (var next in EnumerateSceneGraph(child))
                {
                    yield return next;
                }
            }
        }
        static IEnumerable<INode> EnumerateSceneGraph(INode root)
        {
            if (root == null) yield break;
            yield return root;
            foreach (var next in EnumerateSceneRoot(root))
            {
                yield return next;
            }
        }
        static IEnumerable<IMaterial> EnumerateMaterialNodes(IMaterial node)
        {
            if (node == null) yield break;
            yield return node;
            System.Type type = node.GetType();
            foreach (var property in type.GetProperties())
            {
                if (typeof(IMaterial).IsAssignableFrom(property.PropertyType))
                {
                    foreach (var next in EnumerateMaterialNodes((IMaterial)property.GetValue(node)))
                    {
                        yield return next;
                    }
                }
            }
        }
    }
}
