using Microsoft.Win32;
using RenderToy.DirectX;
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
        public static ICommand CommandSceneNew1 = new RoutedUICommand("New Default Scene 1", "CommandSceneNew1", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandSceneNew2 = new RoutedUICommand("New Default Scene 2", "CommandSceneNew2", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandSceneNew3 = new RoutedUICommand("New Default Scene 3", "CommandSceneNew3", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandSceneOpen = new RoutedUICommand("Open Scene", "CommandSceneLoad", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandScenePlane = new RoutedUICommand("Open Plane", "CommandScenePlane", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandDebugPerformanceTrace = new RoutedUICommand("Performance Trace", "CommandDebugPerformanceTrace", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandDebugPerformanceView = new RoutedUICommand("Performance View", "CommandDebugPerformanceView", typeof(ViewSoftwareCustomizable));
        public static ICommand CommandDocumentOpen = new RoutedUICommand("Open the RenderToy document.", "CommandDocumentOpen", typeof(MainWindow));
        public static ICommand CommandDocumentExport = new RoutedUICommand("Export the RenderToy document to XPS.", "CommandDocumentExport", typeof(MainWindow));
        public static ICommand CommandWindowSoftware = new RoutedUICommand("Open a Software View Window.", "CommandWindowDirectX3DFF", typeof(MainWindow));
        public static ICommand CommandWindowArcturus = new RoutedUICommand("Open an Arcturus View Window.", "CommandWindowArcturus", typeof(MainWindow));
        public static ICommand CommandWindowDirect3D11 = new RoutedUICommand("Open a DirectX 11 View Window.", "CommandWindowDirect3D11", typeof(MainWindow));
        public static ICommand CommandWindowDirect3D12 = new RoutedUICommand("Open a DirectX 12 View Window.", "CommandWindowDirect3D12", typeof(MainWindow));
        public static ICommand CommandWindowTextureLab = new RoutedUICommand("Open a Texture Lab Window.", "CommandWindowTextureLab", typeof(MainWindow));
        public static ICommand CommandStartOpenVR = new RoutedUICommand("Start OpenVR.", "CommandStartOpenVR", typeof(MainWindow));
        void CommandNew()
        {
            var scene = TestScenes.DefaultScene2;
            DataContext = new Document(scene);
            OpenVRPump.Scene = TransformedObject.ConvertToSparseScene(scene);
        }
        public MainWindow()
        {
            Direct3D11Helper.Initialize();
            DXGIHelper.Initialize();
            InitializeComponent();
            CommandBindings.Add(new CommandBinding(CommandSceneNew, (s, e) => {
                CommandNew();
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandSceneNew1, (s, e) =>
            {
                var scene = TestScenes.DefaultScene1;
                DataContext = new Document(scene);
                OpenVRPump.Scene = TransformedObject.ConvertToSparseScene(scene);
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandSceneNew2, (s, e) =>
            {
                var scene = TestScenes.DefaultScene2;
                DataContext = new Document(scene);
                OpenVRPump.Scene = TransformedObject.ConvertToSparseScene(scene);
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandSceneNew3, (s, e) =>
            {
                var scene = TestScenes.DefaultScene3;
                DataContext = new Document(scene);
                OpenVRPump.Scene = TransformedObject.ConvertToSparseScene(scene);
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandSceneOpen, async (s, e) => {
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.Title = "Choose Model File";
                if (ofd.ShowDialog() == true)
                {
                    var scene = new Scene();
                    scene.children.Add(await LoaderModel.LoadFromPathAsync(ofd.FileName));
                    TestScenes.AddOpenVR(scene);
                    DataContext = new Document(scene);
                    OpenVRPump.Scene = TransformedObject.ConvertToSparseScene(scene);
                }
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandScenePlane, (s, e) =>
            {
                var scene = new Scene();
                scene.children.Add(new Node("Plane", new TransformMatrix(Matrix3D.Identity), Plane.Default, StockMaterials.White, StockMaterials.Brick));
                DataContext = new Document(scene);
                OpenVRPump.Scene = TransformedObject.ConvertToSparseScene(scene);
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandDebugPerformanceTrace, (s, e) => {
                CreatePanelDefault(new PerformanceTrace(), "Performance Trace");
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandDebugPerformanceView, (s, e) =>
            {
                CreatePanelDefault(new PerformanceView(), "Performance View");
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowSoftware, (s, e) =>
            {
                var view = new ViewSoftwareCustomizable();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                CreatePanelDefault(view, "RenderToy (Software)");
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowArcturus, (s, e) =>
            {
                var view = new ViewArcturus();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                CreatePanelDefault(view, "RenderToy (Arcturus)");
                e.Handled = true;
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
                CreatePanelDefault(render, "Direct3D11");
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandWindowDirect3D12, (s, e) =>
            {
                var view = new ViewD3D12();
                view.SetBinding(AttachedCamera.CameraProperty, new Binding { Source = FindResource("Camera") });
                view.SetBinding(AttachedView.SceneProperty, new Binding { Path = new PropertyPath("Scene") });
                view.SetBinding(AttachedView.TransformCameraProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformCameraProperty) });
                view.SetBinding(AttachedView.TransformViewProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformViewProperty) });
                view.SetBinding(AttachedView.TransformProjectionProperty, new Binding { Source = FindResource("Camera"), Path = new PropertyPath(Camera.TransformProjectionProperty) });
                CreatePanelDefault(view, "Direct3D12 Render");
                e.Handled = true;
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
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandDocumentOpen, (s, e) =>
            {
                var window = new Window { Title = "RenderToy - A Bit Of History That's Now A Bit Of Silicon..." };
                window.Content = new FlowDocumentReader { Document = new RenderToyDocument() };
                window.Show();
                e.Handled = true;
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
                e.Handled = true;
            }));
            CommandBindings.Add(new CommandBinding(CommandStartOpenVR, (s, e) =>
            {
                #if OPENVR_INSTALLED
                if (DataContext is Document doc)
                {
                    OpenVRHelper.Initialize();
                    OpenVRPump.CreateThread(doc.Scene);
                }
                #endif // OPENVR_INSTALLED
                e.Handled = true;
            }));
            InputBindings.Add(new KeyBinding(CommandSceneNew, Key.N, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandSceneOpen, Key.O, ModifierKeys.Control));
            CommandNew();
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
