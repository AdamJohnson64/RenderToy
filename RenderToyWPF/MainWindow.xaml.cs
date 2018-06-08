////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using Microsoft.Win32;
using RenderToy.Expressions;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Shaders;
using RenderToy.Transforms;
using RenderToy.WPF.Xps;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.IO.Packaging;
using System.Linq;
using System.Windows;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Xps.Packaging;
using System.Windows.Xps.Serialization;

namespace RenderToy.WPF
{
    public partial class MainWindow : Window
    {
        public static RoutedUICommand CommandSceneNew = new RoutedUICommand("New Scene", "CommandSceneNew", typeof(ViewSoftwareCustomizable));
        public static RoutedUICommand CommandSceneOpen = new RoutedUICommand("Open Scene", "CommandSceneLoad", typeof(ViewSoftwareCustomizable));
        public static RoutedUICommand CommandScenePlane = new RoutedUICommand("Open Plane", "CommandScenePlane", typeof(ViewSoftwareCustomizable));
        public static RoutedUICommand CommandSceneAddSphere = new RoutedUICommand("Add Sphere", "CommandSceneAddSphere", typeof(ViewSoftwareCustomizable));
        public static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(ViewSoftwareCustomizable));
        public static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(ViewSoftwareCustomizable));
        public static RoutedUICommand CommandDebugToolPerformanceTrace = new RoutedUICommand("Performance Trace Tool (Debug)", "CommandDebugToolPerformanceTrace", typeof(ViewSoftwareCustomizable));
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
                material.map_bump = new BumpGenerate { U = new MNTexCoordU(), V = new MNTexCoordV(), Displacement = displace };
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
                    try
                    {
                        HLSLExtensions.CompileHLSL(ShaderCode.Text, "vs", "vs_3_0");
                        ShaderErrorsVS.Text = "Vertex Shader Compilation Successful.";
                    }
                    catch (Exception exception)
                    {
                        ShaderErrorsVS.Text = exception.ToString();
                    }
                }
                {
                    try
                    {
                        HLSLExtensions.CompileHLSL(ShaderCode.Text, "ps", "ps_3_0");
                        ShaderErrorsPS.Text = "Pixel Shader Compilation Successful.";
                    }
                    catch (Exception exception)
                    {
                        ShaderErrorsPS.Text = exception.ToString();
                    }
                }
            };
            ShaderCode.Text = HLSL.DX9Full;
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
