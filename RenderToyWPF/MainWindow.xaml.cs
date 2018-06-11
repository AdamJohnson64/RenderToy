////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using Microsoft.Win32;
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
using System.Windows.Controls;
using System.Windows.Data;
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
        public static RoutedUICommand CommandDebugToolPerformanceTrace = new RoutedUICommand("Performance Trace Tool (Debug)", "CommandDebugToolPerformanceTrace", typeof(ViewSoftwareCustomizable));
        public static RoutedUICommand CommandDocumentOpen = new RoutedUICommand("Open the RenderToy document.", "CommandDocumentOpen", typeof(MainWindow));
        public static RoutedUICommand CommandDocumentExport = new RoutedUICommand("Export the RenderToy document to XPS.", "CommandDocumentExport", typeof(MainWindow));
        public static RoutedUICommand CommandWindowSoftware = new RoutedUICommand("Open a Software View Window.", "CommandWindowDirectX3DFF", typeof(MainWindow));
        public static RoutedUICommand CommandWindowDirect3D9FF = new RoutedUICommand("Open a DirectX 9 (Fixed Function) View Window.", "CommandWindowDirectX3DFF", typeof(MainWindow));
        public static RoutedUICommand CommandWindowDirect3D9 = new RoutedUICommand("Open a DirectX 9 View Window.", "CommandWindowDirect3D9", typeof(MainWindow));
        public static RoutedUICommand CommandWindowDirect3D11 = new RoutedUICommand("Open a DirectX 11 View Window.", "CommandWindowDirect3D11", typeof(MainWindow));
        public static RoutedUICommand CommandWindowDirect3D12 = new RoutedUICommand("Open a DirectX 12 View Window.", "CommandWindowDirect3D12", typeof(MainWindow));
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
            CommandBindings.Add(new CommandBinding(CommandDebugToolPerformanceTrace, (s, e) => {
                var window = new Window { Title = "Performance Trace Tool", Content = new PerformanceTrace() };
                window.ShowDialog();
                e.Handled = true;
            }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D9FF, (s, e) =>
            {
                var view = new ViewDirectX9FixedFunction();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                view.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                var window = new Window { Title = "RenderToy (Direct3D9 Fixed Function)", Content = view, Owner = this };
                window.SetBinding(DataContextProperty, new Binding { Source = this, Path = new PropertyPath(DataContextProperty) });
                window.Show();
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowSoftware, (s, e) =>
            {
                var view = new ViewSoftwareCustomizable();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                view.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                var window = new Window { Title = "RenderToy (Software)", Content = view, Owner = this };
                window.SetBinding(DataContextProperty, new Binding { Source = this, Path = new PropertyPath(DataContextProperty) });
                window.Show();
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D9, (s, e) =>
            {
                var grid = new Grid();
                grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(25, GridUnitType.Star) });
                grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(75, GridUnitType.Star) });
                var shader = new ShaderEditor { Text = HLSL.D3D9Standard };
                Grid.SetColumn(shader, 0);
                grid.Children.Add(shader);
                var splitter = new GridSplitter { HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Stretch, Width = 4 };
                Grid.SetColumn(shader, 0);
                grid.Children.Add(splitter);
                var view = new ViewDirectX9();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                view.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                Grid.SetColumn(view, 1);
                grid.Children.Add(view);
                var window = new Window { Title = "RenderToy (Direct3D9)", Content = grid, Owner = this };
                window.SetBinding(DataContextProperty, new Binding { Source = this, Path = new PropertyPath(DataContextProperty) });
                window.Show();
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D11, (s, e) =>
            {
                var grid = new Grid();
                grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(25, GridUnitType.Star) });
                grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(75, GridUnitType.Star) });
                var shader = new ShaderEditor { ProfileVS = "vs_5_0", ProfilePS = "ps_5_0", Text = HLSL.D3D11Standard };
                Grid.SetColumn(shader, 0);
                grid.Children.Add(shader);
                var splitter = new GridSplitter { HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Stretch, Width = 4 };
                Grid.SetColumn(shader, 0);
                grid.Children.Add(splitter);
                var view = new ViewDirectX11();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                view.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                view.SetBinding(ViewDirectX11.VertexShaderProperty, new Binding { Source = shader, Path = new PropertyPath(ShaderEditor.BytecodeVSProperty) });
                view.SetBinding(ViewDirectX11.PixelShaderProperty, new Binding { Source = shader, Path = new PropertyPath(ShaderEditor.BytecodePSProperty) });
                Grid.SetColumn(view, 1);
                grid.Children.Add(view);
                var window = new Window { Title = "RenderToy (Direct3D11)", Content = grid, Owner = this };
                window.SetBinding(DataContextProperty, new Binding { Source = this, Path = new PropertyPath(DataContextProperty) });
                window.Show();
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D12, (s, e) =>
            {
                var view = new ViewDirectX12();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                view.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                var window = new Window { Title = "RenderToy (Direct3D12)", Content = view, Owner = this };
                window.SetBinding(DataContextProperty, new Binding { Source = this, Path = new PropertyPath(DataContextProperty) });
                window.Show();
            }));
            CommandBindings.Add(new CommandBinding(CommandDocumentOpen, (s, e) =>
            {
                var window = new Window { Title = "RenderToy - A Bit Of History That's Now A Bit Of Silicon..." };
                window.Content = new FlowDocumentReader { Document = new RenderToyDocument() };
                window.Show();
            }));
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
                                var flowdocument = new RenderToyDocument();
                                var headerTemplate = (DataTemplate)flowdocument.FindResource("HeaderTemplate");
                                var footerTemplate = (DataTemplate)flowdocument.FindResource("FooterTemplate");
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
            DataContext = Document.Default;
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
