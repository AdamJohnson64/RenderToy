﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using Microsoft.Win32;
using RenderToy.DocumentModel;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Shaders;
using RenderToy.Textures;
using RenderToy.Transforms;
using RenderToy.WPF.Xps;
using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.IO.Packaging;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Xps.Packaging;
using System.Windows.Xps.Serialization;

namespace RenderToy.WPF
{
    public class LoadImageConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var filename = value as string;
            if (filename == null) return null;
            try
            {
                return LoaderImage.LoadFromPath(filename);
            }
            catch (Exception e)
            {
                return e;
            }
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
    public partial class MainWindow : Window, IToolWindowCreator
    {
        public static ICommand CommandSceneNew = new RoutedUICommand("New Scene", "CommandSceneNew", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandSceneOpen = new RoutedUICommand("Open Scene", "CommandSceneLoad", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandScenePlane = new RoutedUICommand("Open Plane", "CommandScenePlane", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandDebugToolPerformanceTrace = new RoutedUICommand("Performance Trace Tool (Debug)", "CommandDebugToolPerformanceTrace", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandDocumentOpen = new RoutedUICommand("Open the RenderToy document.", "CommandDocumentOpen", typeof(MainWindow));
        public static ICommand CommandDocumentExport = new RoutedUICommand("Export the RenderToy document to XPS.", "CommandDocumentExport", typeof(MainWindow));
        public static ICommand CommandWindowSoftware = new RoutedUICommand("Open a Software View Window.", "CommandWindowDirectX3DFF", typeof(MainWindow));
        public static ICommand CommandWindowDirect3D9FF = new RoutedUICommand("Open a DirectX 9 (Fixed Function) View Window.", "CommandWindowDirectX3DFF", typeof(MainWindow));
        public static ICommand CommandWindowDirect3D9 = new RoutedUICommand("Open a DirectX 9 View Window.", "CommandWindowDirect3D9", typeof(MainWindow));
        public static ICommand CommandWindowDirect3D11 = new RoutedUICommand("Open a DirectX 11 View Window.", "CommandWindowDirect3D11", typeof(MainWindow));
        public static ICommand CommandWindowDirect3D12 = new RoutedUICommand("Open a DirectX 12 View Window.", "CommandWindowDirect3D12", typeof(MainWindow));
        public static ICommand CommandWindowTextureLab = new RoutedUICommand("Open a Texture Lab Window.", "CommandWindowTextureLab", typeof(MainWindow));
        public static ICommand CommandStartOpenVR = new RoutedUICommand("Start OpenVR.", "CommandStartOpenVR", typeof(MainWindow));
        public MainWindow()
        {
            InitializeComponent();
            CommandBindings.Add(new CommandBinding(CommandSceneNew, (s, e) => {
                DataContext = Document.Default;
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandSceneOpen, (s, e) => {
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.Title = "Choose Model File";
                if (ofd.ShowDialog() == true)
                {
                    Task.Factory.StartNew(() =>
                    {
                        var scene = new Scene();
                        if (Path.GetExtension(ofd.FileName).ToUpperInvariant() == ".BPT")
                        {
                            var root = new Node(Path.GetFileName(ofd.FileName), new TransformMatrix(Matrix3D.Identity), null, StockMaterials.Black, null);
                            root.children.AddRange(LoaderBPT.LoadFromPath(ofd.FileName));
                            scene.children.Add(root);
                        }
                        else if (Path.GetExtension(ofd.FileName).ToUpperInvariant() == ".OBJ")
                        {
                            var root = new Node(Path.GetFileName(ofd.FileName), new TransformMatrix(Matrix3D.Identity), null, StockMaterials.Black, null);
                            root.children.AddRange(LoaderOBJ.LoadFromPath(ofd.FileName));
                            scene.children.Add(root);
                        }
                        else if (Path.GetExtension(ofd.FileName).ToUpperInvariant() == ".PLY")
                        {
                            var root = new Node(Path.GetFileName(ofd.FileName), new TransformMatrix(Matrix3D.Identity), null, StockMaterials.Black, null);
                            root.children.AddRange(LoaderPLY.LoadFromPath(ofd.FileName));
                            scene.children.Add(root);
                        }
                        TestScenes.AddOpenVR(scene);
                        Dispatcher.Invoke(() => { DataContext = new Document(scene); });
                        return scene;
                    });
                    ;
                }
                e.Handled = true;
            }));
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
                scene.children.Add(new Node("Plane", new TransformMatrix(Matrix3D.Identity), Plane.Default, StockMaterials.White, material));
                DataContext = new Document(scene);
            }));
            CommandBindings.Add(new CommandBinding(CommandDebugToolPerformanceTrace, (s, e) => {
                var window = new Window { Title = "Performance Trace Tool", Content = new PerformanceTrace() };
                window.ShowDialog();
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D9FF, (s, e) =>
            {
                var view = new ViewD3D9FixedFunction();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                view.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                CreatePanelDefault(view, "Direct3D9FF Render");
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
                CreatePanelDefault(view, "RenderToy (Software)");
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D9, (s, e) =>
            {
                var shader = new ShaderEditor { ProfileVS = "vs_3_0", ProfilePS = "ps_3_0", Text = HLSL.D3D9Standard };
                var render = new ViewD3D9();
                render.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                render.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                render.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                render.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                render.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                render.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                render.SetBinding(ViewD3D9.VertexShaderProperty, new Binding { Source = shader, Path = new PropertyPath(ShaderEditor.BytecodeVSProperty) });
                render.SetBinding(ViewD3D9.PixelShaderProperty, new Binding { Source = shader, Path = new PropertyPath(ShaderEditor.BytecodePSProperty) });
                CreatePanelNavigation("Direct3D9", render, "Direct3D9 Render", shader, "Direct3D9 Shader");
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D11, (s, e) =>
            {
                var shader = new ShaderEditor { ProfileVS = "vs_5_0", ProfilePS = "ps_5_0", Text = HLSL.D3D11Standard };
                var render = new ViewD3D11();
                render.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                render.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                render.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                render.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                render.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                render.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                render.SetBinding(ViewD3D11.VertexShaderProperty, new Binding { Source = shader, Path = new PropertyPath(ShaderEditor.BytecodeVSProperty) });
                render.SetBinding(ViewD3D11.PixelShaderProperty, new Binding { Source = shader, Path = new PropertyPath(ShaderEditor.BytecodePSProperty) });
                CreatePanelNavigation("Direct3D11", render, "Direct3D11 Render", shader, "Direct3D11 Shader");
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D12, (s, e) =>
            {
                var view = new ViewD3D12();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                view.SetBinding(AttachedView.TransformModelViewProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformModelViewProjectionProperty) });
                CreatePanelDefault(view, "Direct3D12 Render");
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowTextureLab, (s, e) =>
            {
                var browser = new ListBox();
                var rootassembly = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
                var rootassets = Path.Combine(rootassembly, "..\\..\\ThirdParty\\RenderToyAssets");
                var files = Directory.EnumerateFiles(rootassets, "*", SearchOption.AllDirectories)
                    .Where(i => new[] { ".HDR", ".PNG", ".TGA" }.Contains(Path.GetExtension(i).ToUpperInvariant()));
                browser.SetBinding(ListBox.ItemsSourceProperty, new Binding { Source = files });
                var imagezoom = new ViewZoom();
                var image = new ViewMaterial();
                image.SetBinding(ViewMaterial.MaterialSourceProperty, new Binding { Source = browser, Path = new PropertyPath(ListBox.SelectedItemProperty), Converter = new LoadImageConverter() });
                imagezoom.Content = image;
                CreatePanelNavigation("Texture Lab", imagezoom, "Image", browser, "Browser");
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
                                paginator.PageSize = new Size(1024, 1280);
                                serialization.SaveAsXaml(paginator);
                                serialization.Commit();
                            }
                        }
                    }
                }
            }));
            CommandBindings.Add(new CommandBinding(CommandStartOpenVR, (s, e) =>
            {
                #if OPENVR_INSTALLED
                if (DataContext is Document doc)
                {
                    OpenVRPump.CreateThread(doc.Scene);
                }
                #endif // OPENVR_INSTALLED
            }));
            InputBindings.Add(new KeyBinding(CommandSceneNew, Key.N, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandSceneOpen, Key.O, ModifierKeys.Control));
            Task.Factory.StartNew(() =>
            {
                var initialscene = Document.Default;
                Dispatcher.Invoke(() => { DataContext = Document.Default; });
            });
        }
        void CreatePanelDefault(FrameworkElement control, string title)
        {
            var tabitem = new TabItem { Header = title, Content = control };
            TabControlMain.Items.Add(tabitem);
            TabControlMain.SelectedItem = tabitem;
        }
        void CreatePanelNavigation(string title, FrameworkElement controlmain, string titlemain, FrameworkElement controlnavigation, string titlenavigation)
        {
            var grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(192, GridUnitType.Pixel) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(4, GridUnitType.Pixel) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(100, GridUnitType.Star) });
            var tabcontrolleft = new TabControl();
            tabcontrolleft.Items.Add(new TabItem { Header = titlenavigation, Content = controlnavigation });
            Grid.SetColumn(tabcontrolleft, 0);
            grid.Children.Add(tabcontrolleft);
            var splitter = new GridSplitter { HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Stretch, Width = 4 };
            Grid.SetColumn(splitter, 1);
            grid.Children.Add(splitter);
            TabControl tabcontrolmain = new TabControl();
            tabcontrolmain.Items.Add(new TabItem { Header = titlemain, Content = controlmain });
            Grid.SetColumn(tabcontrolmain, 2);
            grid.Children.Add(tabcontrolmain);
            var tabitem = new TabItem { Content = grid, Header = title };
            TabControlMain.Items.Add(tabitem);
            TabControlMain.SelectedItem = tabitem;
        }
        public void CreateToolWindow(object content)
        {
            var newitemscontainer = new TabControl();
            newitemscontainer.Items.Add(content);
            var window = new Window { Content = newitemscontainer, Title = "Tool Window", Width = 256, Height = 256 };
            window.DataContext = DataContext;
            window.Show();
        }
    }
    class Document
    {
        public SparseScene Scene { get; private set; }
        public ObservableCollection<IMaterial> Materials { get; private set; }
        public static Document Default = new Document(TestScenes.DefaultScene);
        public Document(IScene scene)
        {
            Scene = TransformedObject.ConvertToSparseScene(scene);
            var materials = Scene
                .Select(i => i.NodeMaterial)
                .OfType<IMaterial>()
                .Distinct();
            //.SelectMany(i => EnumerateNodes(i));
            Materials = new ObservableCollection<IMaterial>(materials);
        }
    }
}
