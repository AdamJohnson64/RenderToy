using RenderToy.Cameras;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Primitives;
using RenderToy.RenderControl;
using RenderToy.RenderMode;
using RenderToy.SceneGraph;
using RenderToy.Transforms;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class ViewSoftwareCustomizable : FrameworkElement
    {
        static RoutedUICommand CommandResolution100 = new RoutedUICommand("100% Resolution", "CommandResolution100", typeof(ViewSoftwareCustomizable));
        static RoutedUICommand CommandResolution50 = new RoutedUICommand("50% Resolution", "CommandResolution50", typeof(ViewSoftwareCustomizable));
        static RoutedUICommand CommandResolution25 = new RoutedUICommand("25% Resolution", "CommandResolution25", typeof(ViewSoftwareCustomizable));
        static RoutedUICommand CommandResolution10 = new RoutedUICommand("10% Resolution", "CommandResolution10", typeof(ViewSoftwareCustomizable));
        static ViewSoftwareCustomizable()
        {
            AttachedView.SceneProperty.OverrideMetadata(typeof(ViewSoftwareCustomizable), new FrameworkPropertyMetadata(null, (s, e) => ((ViewSoftwareCustomizable)s).InvalidateVisual()));
            AttachedView.TransformViewProperty.OverrideMetadata(typeof(ViewSoftwareCustomizable), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) => ((ViewSoftwareCustomizable)s).InvalidateVisual()));
            AttachedView.TransformProjectionProperty.OverrideMetadata(typeof(ViewSoftwareCustomizable), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) => ((ViewSoftwareCustomizable)s).InvalidateVisual()));
        }
        public ViewSoftwareCustomizable()
        {
            RenderCall = RenderCallCommands.Calls[0];
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
            Focusable = true;
            CommandBindings.Add(new CommandBinding(CommandResolution100, (s, e) => { RenderResolution = 1; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandResolution50, (s, e) => { RenderResolution = 2; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandResolution25, (s, e) => { RenderResolution = 4; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandResolution10, (s, e) => { RenderResolution = 10; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            // Generate commands for render modes.
            foreach (var call in RenderCallCommands.Calls)
            {
                CommandBindings.Add(new CommandBinding(RenderCallCommands.Commands[call], (s, e) => { RenderCall = call; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            }
            // Generate context menu.
            var menu = new ContextMenu { LayoutTransform = new ScaleTransform(1.5, 1.5) };
            {
                var menu_group = new MenuItem { Header = "Resolution" };
                menu_group.Items.Add(new MenuItem { Command = CommandResolution100 });
                menu_group.Items.Add(new MenuItem { Command = CommandResolution50 });
                menu_group.Items.Add(new MenuItem { Command = CommandResolution25 });
                menu_group.Items.Add(new MenuItem { Command = CommandResolution10 });
                menu.Items.Add(menu_group);
            }
            var scenehierarchy = new Scene();
            scenehierarchy.children.Add(new Node("Sphere (Red)", new TransformMatrix(MathHelp.CreateMatrixIdentity()), Sphere.Default, StockMaterials.Red, StockMaterials.PlasticRed));
            var scene = TransformedObject.Enumerate(scenehierarchy);
            foreach (var group in RenderCallCommands.Calls.GroupBy(x => RenderCall.GetDisplayNameBare(x.MethodInfo.Name)))
            {
                var menu_group = new MenuItem { Header = group.Key };
                Matrix3D mvp = MathHelp.Invert(MathHelp.CreateMatrixLookAt(new Vector3D(0, 0, -2), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0)));
                mvp = MathHelp.Multiply(mvp, Perspective.CreateProjection(0.01, 100.0, 60.0 * System.Math.PI / 180.0, 60.0 * System.Math.PI / 180.0));
                foreach (var call in group)
                {
                    WriteableBitmap bitmap = new WriteableBitmap(64, 64, 0, 0, PixelFormats.Bgra32, null);
                    bitmap.Lock();
                    call.Action(scene, mvp, bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride, null);
                    bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));
                    bitmap.Unlock();
                    menu_group.Items.Add(new MenuItem { Command = RenderCallCommands.Commands[call], Icon = new Image { Source = bitmap } });
                }
                menu.Items.Add(menu_group);
            }
            this.ContextMenu = menu;
        }
        #region - Section : RenderMode Option -
        RenderCall RenderCall
        {
            set
            {
                renderCall = value;
                if (!renderCall.IsMultipass)
                {
                    renderMode = new SinglePassAsyncAdaptor(renderCall, () => Dispatcher.Invoke(InvalidateVisual));
                }
                else
                {
                    renderMode = new MultiPassAsyncAdaptor(renderCall, () => Dispatcher.Invoke(InvalidateVisual));
                }
                renderMode.SetScene(AttachedView.GetScene(this));
                InvalidateVisual();
            }
        }
        RenderCall renderCall;
        IMultiPass RenderMode
        {
            get
            {
                return renderMode;
            }
        }
        IMultiPass renderMode;
        #endregion
        #region - Section : RenderResolution Option -
        int RenderResolution
        {
            get { return renderResolution; }
            set { renderResolution = value; InvalidateVisual(); }
        }
        int renderResolution = 2;
        #endregion
        #region - Overrides : RenderViewportBase -
        protected override void OnPropertyChanged(DependencyPropertyChangedEventArgs e)
        {
            base.OnPropertyChanged(e);
            if (e.Property == AttachedView.SceneProperty)
            {
                if (RenderMode == null) return;
                RenderMode.SetScene((IEnumerable<TransformedObject>)e.NewValue);
            }
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            if (RenderMode == null) return;
            int RENDER_WIDTH = (int)System.Math.Ceiling(ActualWidth) / RenderResolution;
            int RENDER_HEIGHT = (int)System.Math.Ceiling(ActualHeight) / RenderResolution;
            if (RENDER_WIDTH == 0 || RENDER_HEIGHT == 0) return;
            WriteableBitmap bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT, 0, 0, PixelFormats.Bgra32, null);
            RenderMode.SetCamera(AttachedView.GetTransformView(this) * AttachedView.GetTransformProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
            RenderMode.SetTarget(bitmap.PixelWidth, bitmap.PixelHeight);
            bitmap.Lock();
            RenderMode.CopyTo(bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));
            bitmap.Unlock();
            drawingContext.DrawImage(bitmap, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        #endregion
        #region - Overrides : UIElement -
        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            base.OnMouseDown(e);
            Focus();
        }
        #endregion
    }
    class RenderCallCommands
    {
        static RenderCallCommands()
        {
            Calls = RenderCall.Generate(new[] { typeof(RenderModeCS), typeof(RenderToyCLI) }).ToArray();
            Commands = Calls.ToDictionary(x => x, y => new RoutedUICommand(RenderCall.GetDisplayNameFull(y.MethodInfo.Name), y.MethodInfo.Name, typeof(RenderCallCommands)));
        }
        public static readonly RenderCall[] Calls;
        public static readonly Dictionary<RenderCall, RoutedUICommand> Commands;
    }
}