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
using System.IO;
using System.IO.Packaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Xps.Packaging;
using System.Windows.Xps.Serialization;

namespace RenderToy.WPF
{
    public class DocumentPaginatorWrapper : DocumentPaginator
    {
        public static DependencyProperty PageNumberProperty = DependencyProperty.RegisterAttached("PageNumber", typeof(int), typeof(DocumentPaginatorWrapper), new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.AffectsArrange | FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.Inherits));
        public DocumentPaginatorWrapper(DocumentPaginator paginator, double header, double footer, DataTemplate templateHeader, DataTemplate templateFooter)
        {
            heightHeader = header;
            heightFooter = footer;
            innerPaginator = paginator;
            this.templateHeader = templateHeader;
            this.templateFooter = templateFooter;
        }
        public override DocumentPage GetPage(int pageNumber)
        {
            DocumentPage page = null;
            page = (this == innerPaginator) ? new DocumentPage(new ContainerVisual()) : innerPaginator.GetPage(pageNumber);
            var newpage = new ContainerVisual();
            newpage.SetValue(PageNumberProperty, pageNumber);
            {
                var sectiondocument = (FlowDocument)templateHeader.LoadContent();
                sectiondocument.ColumnWidth = PageSize.Width;
                var sectionpaginator = ((IDocumentPaginatorSource)sectiondocument).DocumentPaginator;
                sectionpaginator.PageSize = PageSize;
                var sectionpage = sectionpaginator.GetPage(0);
                var sectioncontainer = new ContainerVisual();
                sectioncontainer.Offset = new Vector(0, 0);
                sectioncontainer.Children.Add(sectionpage.Visual);
                newpage.Children.Add(sectioncontainer);
            }
            {
                var sectioncontainer = new ContainerVisual();
                sectioncontainer.Children.Add(page.Visual);
                sectioncontainer.Offset = new Vector(0, heightHeader);
                newpage.Children.Add(sectioncontainer);
            }
            {
                var sectiondocument = (FlowDocument)templateFooter.LoadContent();
                sectiondocument.SetValue(PageNumberProperty, pageNumber);
                sectiondocument.ColumnWidth = PageSize.Width;
                var sectionpaginator = ((IDocumentPaginatorSource)sectiondocument).DocumentPaginator;
                sectionpaginator.PageSize = PageSize;
                var sectionpage = sectionpaginator.GetPage(0);
                var sectioncontainer = new ContainerVisual();
                sectioncontainer.Offset = new Vector(0, heightHeader + page.Size.Height);
                sectioncontainer.Children.Add(sectionpage.Visual);
                newpage.Children.Add(sectioncontainer);
            }
            return new DocumentPage(newpage, PageSize, page.BleedBox, page.ContentBox);
        }
        public override bool IsPageCountValid { get { return innerPaginator.IsPageCountValid; } }
        public override int PageCount { get { return innerPaginator.PageCount; } }
        public override Size PageSize
        {
            get
            {
                var size = innerPaginator.PageSize;
                size.Height = size.Height + heightHeader + heightFooter;
                return size;
            }
            set
            {
                if (this != innerPaginator)
                {
                    innerPaginator.PageSize = value;
                }
            }
        }
        public override IDocumentPaginatorSource Source { get { return innerPaginator.Source; } }
        DataTemplate templateHeader;
        DataTemplate templateFooter;
        double heightHeader;
        double heightFooter;
        DocumentPaginator innerPaginator;
    }
    public partial class MainWindow : Window
    {
        public static RoutedUICommand CommandSceneNew = new RoutedUICommand("New Scene", "CommandSceneNew", typeof(View3DUser));
        public static RoutedUICommand CommandSceneOpen = new RoutedUICommand("Open Scene", "CommandSceneLoad", typeof(View3DUser));
        public static RoutedUICommand CommandScenePlane = new RoutedUICommand("Open Plane", "CommandScenePlane", typeof(View3DUser));
        public static RoutedUICommand CommandSceneAddSphere = new RoutedUICommand("Add Sphere", "CommandSceneAddSphere", typeof(View3DUser));
        public static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(View3DUser));
        public static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(View3DUser));
        public static RoutedUICommand CommandDebugToolPerformanceTrace = new RoutedUICommand("Performance Trace Tool (Debug)", "CommandDebugToolPerformanceTrace", typeof(View3DUser));
        public static RoutedUICommand CommandDocumentExport = new RoutedUICommand("Export the RenderToy document to XPS.", "CommandDocumentExport", typeof(MainWindow));
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
                        var root = new Node(Path.GetFileName(ofd.FileName), new TransformMatrix(Matrix3D.Identity), null, StockMaterials.Black, null);
                        foreach (var primitive in LoaderBPT.LoadFromPath(ofd.FileName))
                        {
                            root.children.Add(new Node("Bezier Patch", new TransformMatrix(Matrix3D.Identity), primitive, StockMaterials.White, StockMaterials.PlasticWhite));
                        }
                        scene.children.Add(root);
                    }
                    else if (Path.GetExtension(ofd.FileName).ToUpperInvariant() == ".OBJ")
                    {
                        foreach (var node in LoaderOBJ.LoadFromPath(ofd.FileName))
                        {
                            scene.children.Add(node);
                        }
                    }
                    else if (Path.GetExtension(ofd.FileName).ToUpperInvariant() == ".PLY")
                    {
                        scene.children.Add(new Node(Path.GetFileName(ofd.FileName), new TransformMatrix(MathHelp.CreateMatrixScale(100, 100, 100)), LoaderPLY.LoadFromPath(ofd.FileName), StockMaterials.White, StockMaterials.PlasticWhite));
                    }
                    DataContext = new Document(scene);
                }
                e.Handled = true;
            }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandScenePlane, (s, e) =>
            {
                var scene = new Scene();
                var material = new LoaderOBJ.OBJMaterial();
                material.map_Kd = StockMaterials.Brick;
                var displace = new MNSubtract {
                    Lhs = new BrickMask { U = new MNMultiply { Lhs = new MNTexCoordU(), Rhs = new MNConstant { Value = 4 } }, V = new MNMultiply { Lhs = new MNTexCoordV(), Rhs = new MNConstant { Value = 4 } } },
                    Rhs = new MNMultiply { Lhs = new Perlin2D { U = new MNMultiply { Lhs = new MNTexCoordU(), Rhs = new MNConstant { Value = 512 } }, V = new MNMultiply { Lhs = new MNTexCoordV(), Rhs = new MNConstant { Value = 512 } } }, Rhs = new MNConstant { Value = 0.001 } }
                };
                material.map_bump = new BumpGenerate { Displacement = displace };
                material.displacement = new MNVector4D { R = displace, G = displace, B = displace, A = new MNConstant { Value = 1 } };
                scene.children.Add(new Node("Plane", new TransformMatrix(Matrix3D.Identity), new Plane(), StockMaterials.White, material));
                DataContext = new Document(scene);
            }));
            CommandBindings.Add(new CommandBinding(CommandSceneAddSphere, (s, e) => 
            {
                var root = ((Document)DataContext).Scene as Scene;
                if (root == null) return;
                root.children.Add(new Node("Plane", new TransformMatrix(Matrix3D.Identity), new Sphere(), StockMaterials.White, StockMaterials.Brick));
            }));
            CommandBindings.Add(new CommandBinding(CommandRenderPreviewsToggle, (s, e) => { ViewPreview.Visibility = ViewPreview.Visibility == Visibility.Hidden ? Visibility.Visible : Visibility.Hidden; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframeToggle, (s, e) => { ViewWireframe.Visibility = ViewWireframe.Visibility == Visibility.Hidden ? Visibility.Visible : Visibility.Hidden; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandDebugToolPerformanceTrace, (s, e) => {
                var window = new Window { Title = "Performance Trace Tool", Content = new PerformanceTrace() };
                window.ShowDialog();
                e.Handled = true;
            }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandDocumentExport, (s, e) =>
            {
                var savefiledialog = new SaveFileDialog();
                savefiledialog.Filter = "*.XPS|XPS Document";
                savefiledialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                if (savefiledialog.ShowDialog() == true)
                {
                    string xpsfilename = savefiledialog.FileName;
                    if (String.IsNullOrWhiteSpace(Path.GetExtension(xpsfilename)))
                    {
                        xpsfilename = Path.ChangeExtension(xpsfilename, "xps");
                    }
                    using (var stream = File.Open(xpsfilename, FileMode.Create))
                    {
                        using (var package = Package.Open(stream, FileMode.Create, FileAccess.ReadWrite))
                        {
                            using (var xpsdocument = new XpsDocument(package, CompressionOption.Maximum))
                            {
                                var flowdocument = (FlowDocument)Application.Current.Resources["RenderToyDocument"];
                                var headerTemplate = (DataTemplate)flowdocument.Resources["HeaderTemplate"];
                                var footerTemplate = (DataTemplate)flowdocument.Resources["FooterTemplate"];
                                var serialization = new XpsSerializationManager(new XpsPackagingPolicy(xpsdocument), false);
                                var paginator = new DocumentPaginatorWrapper(((IDocumentPaginatorSource)flowdocument).DocumentPaginator, 100, 100, headerTemplate, footerTemplate);
                                paginator.PageSize = new System.Windows.Size(1024, 1280);
                                serialization.SaveAsXaml(paginator);
                                serialization.Commit();
                            }
                        }
                    }
                }
            }));
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
@"float4x4 TransformCamera : register(c0);
float4x4 TransformModel : register(c4);
float4x4 TransformView : register(c8);
float4x4 TransformProjection : register(c12);
float4x4 TransformModelViewProjection : register(c16);
sampler2D SamplerAlbedo : register(s0);
sampler2D SamplerMask : register(s1);
sampler2D SamplerBump : register(s2);
sampler2D SamplerDisplacement : register(s3);

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
    float3 EyeVector : TEXCOORD1;
};

VS_OUTPUT vs(VS_INPUT input) {
    VS_OUTPUT result;
    result.Position = mul(TransformModelViewProjection, input.Position);
    result.Normal = input.Normal;
    result.TexCoord = input.TexCoord;
    result.Color = input.Color;
    result.Tangent = input.Tangent;
    result.Bitangent = input.Bitangent;
    result.EyeVector = float3(TransformCamera[0].w, TransformCamera[1].w, TransformCamera[2].w) - input.Position.xyz;
    return result;
}

float4 ps(VS_OUTPUT input) : SV_Target {
    ////////////////////////////////////////////////////////////////////////////////
    // Stencil Mask
    if (tex2D(SamplerMask, input.TexCoord).r < 0.5) discard;

    ////////////////////////////////////////////////////////////////////////////////
    // Reconstruct Tangent Basis
    float3x3 tbn = {input.Tangent, input.Bitangent, input.Normal};

    ////////////////////////////////////////////////////////////////////////////////
    // Displacement Mapping (Steep Parallax)
    float height = 1.0;
    float bumpScale = 0.02;
    float numSteps = 20;
    float2 offsetCoord = input.TexCoord.xy;
    float sampledHeight = tex2D(SamplerDisplacement, offsetCoord).r;
    float3 tangentSpaceEye = mul(input.EyeVector, transpose(tbn));
    numSteps = lerp(numSteps * 2, numSteps, normalize(tangentSpaceEye).z);
    float step = 1.0 / numSteps;
    float2 delta = -float2(tangentSpaceEye.x, tangentSpaceEye.y) * bumpScale / (tangentSpaceEye.z * numSteps);
    int maxiter = 50;
    int iter = 0;
    while (sampledHeight < height && iter < maxiter) {
        height -= step;
        offsetCoord += delta;
        sampledHeight = tex2D(SamplerDisplacement, offsetCoord).r;
        ++iter;
    }
    height = sampledHeight;

    ////////////////////////////////////////////////////////////////////////////////
    // Bump Mapping Normal
    float3 bump = normalize(tex2D(SamplerBump, offsetCoord).rgb * 2 - 1);
    float3 normal = mul(bump, tbn);

    ////////////////////////////////////////////////////////////////////////////////
    // Simple Lighting
    float light = clamp(dot(normal, normalize(float3(1,1,1))), 0, 1);

    ////////////////////////////////////////////////////////////////////////////////
    // Final Color
    return float4(light * tex2D(SamplerAlbedo, offsetCoord).rgb, 1);
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
                .Select(i => i.Material)
                .OfType<IMaterial>()
                .Distinct();
            //.SelectMany(i => EnumerateNodes(i));
            Materials = new ObservableCollection<IMaterial>(materials);
        }
        static IEnumerable<INode> EnumerateSceneRoot(IScene root)
        {
            if (root == null) yield break;
            foreach (var child in root.Children)
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
