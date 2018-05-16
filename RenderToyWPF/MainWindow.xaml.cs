////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using Microsoft.Win32;
using RenderToy.Materials;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Transforms;
using RenderToy.Utility;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;

namespace RenderToy.WPF
{
    public partial class MainWindow : Window
    {
        public static RoutedUICommand CommandSceneNew = new RoutedUICommand("New Scene", "CommandSceneNew", typeof(View3DUser));
        public static RoutedUICommand CommandSceneOpen = new RoutedUICommand("Open Scene (PLY)", "CommandSceneLoad", typeof(View3DUser));
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
                    scene.AddChild(new Node("Plane (Ground)", new TransformMatrix(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), StockMaterials.LightGray, StockMaterials.MarbleTile()));
                    scene.AddChild(new Node(Path.GetFileName(ofd.FileName), new TransformMatrix(MathHelp.CreateMatrixScale(100, 100, 100)), LoaderPLY.LoadFromPath(ofd.FileName), StockMaterials.LightGray, StockMaterials.PlasticRed));
                    DataContext = new Document(scene);
                }
                e.Handled = true;
            }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
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
                .OfType<IMaterial>();
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
