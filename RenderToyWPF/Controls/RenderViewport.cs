////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy
{
    public abstract class RenderViewportBase : FrameworkElement
    {
        #region - Section : Scene -
        protected Scene Scene {
            get
            {
                return scene;
            }
            set
            {
                scene = value;
                InvalidateScene();
            }
        }
        Scene scene = Scene.Default;
        void InvalidateScene()
        {
            if (SceneChanged != null) SceneChanged();
            InvalidateVisual();
        }
        public delegate void SceneChangedHandler();
        public event SceneChangedHandler SceneChanged;
        #endregion
        #region - Section : Camera -
        protected Matrix3D View
        {
            get
            {
                return MathHelp.Invert(Camera.Transform);
            }
        }
        protected Matrix3D Projection
        {
            get
            {
                return CameraMat.Projection;
            }
        }
        protected Matrix3D MVP
        {
            get
            {
                return View * Projection * CameraPerspective.AspectCorrectFit(ActualWidth, ActualHeight);
            }
        }
        TransformPosQuat Camera = new TransformPosQuat { Position = new Point3D(0, 2, -5) };
        CameraPerspective CameraMat = new CameraPerspective();
        void InvalidateCamera()
        {
            if (CameraChanged != null) CameraChanged();
            InvalidateVisual();
        }
        public delegate void CameraChangedHandler();
        public event CameraChangedHandler CameraChanged;
        #endregion
        #region - Section : Rendering -
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            OnRenderToy(drawingContext);
        }
        protected abstract void OnRenderToy(DrawingContext drawingContext);
        #endregion
        #region - Overrides : UIElement -
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            CaptureMouse();
            Mouse.OverrideCursor = Cursors.None;
            isDragging = true;
            dragFrom = System.Windows.Forms.Cursor.Position;
        }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            Mouse.OverrideCursor = null;
            ReleaseMouseCapture();
            isDragging = false;
            InvalidateVisual();
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!isDragging) return;
            System.Drawing.Point dragTo = System.Windows.Forms.Cursor.Position;
            double dx = dragTo.X - dragFrom.X;
            double dy = dragTo.Y - dragFrom.Y;
            System.Windows.Forms.Cursor.Position = dragFrom;
            // Detect modifier keys.
            bool isPressedLeftControl = Keyboard.IsKeyDown(Key.LeftCtrl);
            bool isPressedLeftShift = Keyboard.IsKeyDown(Key.LeftShift);
            // Process camera motion with modifier keys.
            if (isPressedLeftShift && isPressedLeftControl)
            {
                // Truck Mode (CTRL + SHIFT).
                Camera.TranslatePost(new Vector3D(0, 0, dy * -0.05));
                InvalidateCamera();
            }
            else if (!isPressedLeftShift && isPressedLeftControl)
            {
                // Rotate Mode (CTRL Only)
                Camera.RotatePre(new Quaternion(new Vector3D(0, 1, 0), dx * 0.05));
                Camera.RotatePost(new Quaternion(new Vector3D(1, 0, 0), dy * 0.05));
                InvalidateCamera();
            }
            else if (!isPressedLeftShift && !isPressedLeftControl)
            {
                // Translation Mode (no modifier keys).
                Camera.TranslatePost(new Vector3D(dx * -0.05, dy * 0.05, 0));
                InvalidateCamera();
            }
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            Camera.TranslatePost(new Vector3D(0, 0, e.Delta * 0.01));
            InvalidateCamera();
        }
        bool isDragging = false;
        System.Drawing.Point dragFrom;
        #endregion
    }
    class RenderViewport : RenderViewportBase
    {
        static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(RenderViewport));
        static RoutedUICommand CommandResolution100 = new RoutedUICommand("100% Resolution", "CommandResolution100", typeof(RenderViewport));
        static RoutedUICommand CommandResolution50 = new RoutedUICommand("50% Resolution", "CommandResolution50", typeof(RenderViewport));
        static RoutedUICommand CommandResolution25 = new RoutedUICommand("25% Resolution", "CommandResolution25", typeof(RenderViewport));
        static RoutedUICommand CommandResolution10 = new RoutedUICommand("10% Resolution", "CommandResolution10", typeof(RenderViewport));
        static RoutedUICommand CommandSceneNew = new RoutedUICommand("New Scene", "CommandSceneNew", typeof(RenderViewport));
        static RoutedUICommand CommandSceneLoad = new RoutedUICommand("Load Scene (PLY)", "CommandSceneLoad", typeof(RenderViewport));
        public RenderViewport()
        {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
            Focusable = true;
            CommandBindings.Add(new CommandBinding(CommandRenderPreviewsToggle, (s, e) => { RenderPreviews = !RenderPreviews; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframeToggle, (s, e) => { RenderWireframe = !RenderWireframe; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandResolution100, (s, e) => { RenderResolution = 1; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandResolution50, (s, e) => { RenderResolution = 2; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandResolution25, (s, e) => { RenderResolution = 4; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandResolution10, (s, e) => { RenderResolution = 10; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandSceneNew, (s, e) => { Scene = Scene.Default; e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandSceneLoad, (s, e) => {
                OpenFileDialog ofd = new OpenFileDialog();
                ofd.Title = "Choose Model File";
                if (ofd.ShowDialog() == true)
                {
                    Scene scene = new Scene();
                    scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), Materials.LightGray, new CheckerboardMaterial(Materials.Black, Materials.White)));
                    scene.AddChild(new Node(new TransformMatrix3D(MathHelp.CreateMatrixScale(100, 100, 100)), MeshPLY.LoadFromPath(ofd.FileName), Materials.LightGray, Materials.PlasticRed));
                    Scene = scene;
                }
                e.Handled = true;
            }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            // Define input bindings for common display modes.
            InputBindings.Add(new KeyBinding(CommandRenderPreviewsToggle, Key.P, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframeToggle, Key.W, ModifierKeys.Control));
            // Generate commands for render modes.
            foreach (var call in RenderCallCommands.Calls)
            {
                CommandBindings.Add(new CommandBinding(RenderCallCommands.Commands[call], (s, e) => { RenderMode = MultiPass.Create(call); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            }
            // Generate context menu.
            var menu = new ContextMenu();
            {
                var menu_group = new MenuItem { Header = "File" };
                menu_group.Items.Add(new MenuItem { Command = CommandSceneNew });
                menu_group.Items.Add(new MenuItem { Command = CommandSceneLoad });
                menu.Items.Add(menu_group);
            }
            {
                var menu_group = new MenuItem { Header = "Overlays" };
                menu_group.Items.Add(new MenuItem { Command = CommandRenderPreviewsToggle });
                menu_group.Items.Add(new MenuItem { Command = CommandRenderWireframeToggle });
                menu.Items.Add(menu_group);
            }
            {
                var menu_group = new MenuItem { Header = "Resolution" };
                menu_group.Items.Add(new MenuItem { Command = CommandResolution100 });
                menu_group.Items.Add(new MenuItem { Command = CommandResolution50 });
                menu_group.Items.Add(new MenuItem { Command = CommandResolution25 });
                menu_group.Items.Add(new MenuItem { Command = CommandResolution10 });
                menu.Items.Add(menu_group);
            }
            foreach (var group in RenderCallCommands.Calls.GroupBy(x => RenderCall.GetDisplayNameBare(x.MethodInfo.Name)))
            {
                var menu_group = new MenuItem { Header = group.Key };
                foreach (var call in group)
                {
                    menu_group.Items.Add(new MenuItem { Command = RenderCallCommands.Commands[call] });
                }
                menu.Items.Add(menu_group);
            }
            this.ContextMenu = menu;
            // HACK: This causes a repaint every 10ms for multipass renders.
            multipassRedraw = new System.Windows.Forms.Timer();
            multipassRedraw.Interval = 10;
            multipassRedraw.Tick += (s, e) =>
            {
                if (renderAgain)
                {
                    InvalidateVisual();
                }
            };
            multipassRedraw.Start();
        }
        System.Windows.Forms.Timer multipassRedraw;
        #region - Section : RenderMode Option -
        MultiPass RenderMode
        {
            get { return renderMode; }
            set { renderMode = value; InvalidateVisual(); }
        }
        MultiPass renderMode = MultiPass.Create(RenderCallCommands.Calls[0]);
        #endregion
        #region - Section : RenderResolution Option -
        int RenderResolution
        {
            get { return renderResolution; }
            set { renderResolution = value; InvalidateVisual(); }
        }
        int renderResolution = 2;
        #endregion
        #region - Section : RenderPreviews Option -
        bool RenderPreviews
        {
            get { return renderPreviews; }
            set { renderPreviews = value; InvalidateVisual(); }
        }
        bool renderPreviews = true;
        #endregion
        #region - Section : RenderWireframe Option -
        bool RenderWireframe
        {
            get { return renderWireframe; }
            set { renderWireframe = value; InvalidateVisual(); }
        }
        bool renderWireframe = false;
        #endregion
        #region - Overrides : RenderViewportBase -
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            if (RenderMode != null)
            {
                int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth) / RenderResolution;
                int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight) / RenderResolution;
                RenderMode.SetScene(Scene);
                RenderMode.SetCamera(MVP);
                RenderMode.SetTarget(RENDER_WIDTH, RENDER_HEIGHT);
                WriteableBitmap bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT, 0, 0, PixelFormats.Bgra32, null);
                bitmap.Lock();
                renderAgain = RenderMode.CopyTo(bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
                bitmap.AddDirtyRect(new Int32Rect(0, 0, RENDER_WIDTH, RENDER_HEIGHT));
                bitmap.Unlock();
                drawingContext.DrawImage(bitmap, new Rect(0, 0, ActualWidth, ActualHeight));
            }
            if (RenderWireframe)
            {
                drawingContext.PushOpacity(0.5);
                drawingContext.DrawImage(ImageHelp.CreateImage(RenderCS.WireframeCPUF64, Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                drawingContext.Pop();
            }
            if (RenderPreviews)
            {
                Action<RenderCall.FillFunction, int, int, int> drawpreview = (fillwith, stacky, render_width, render_height) =>
                {
                    double frame_l = ActualWidth - 128 - 8;
                    double frame_t = 8 + (96 + 8) * stacky;
                    double frame_w = 128;
                    double frame_h = 96;
                    double image_l = frame_l + 8;
                    double image_t = frame_t + 8;
                    double image_w = frame_w - 8 * 2;
                    double image_h = frame_h - 8 * 2;
                    drawingContext.DrawRoundedRectangle(Brushes.White, new Pen(Brushes.DarkGray, 2), new Rect(frame_l, frame_t, frame_w, frame_h), 8, 8);
                    var imagesource = ImageHelp.CreateImage(fillwith, Scene, View * Projection * CameraPerspective.AspectCorrectFit(image_w, image_h), render_width, render_height);
                    drawingContext.DrawImage(imagesource, new Rect(image_l, image_t, image_w, image_h));
                };
                drawpreview(RenderCS.PointCPUF64, 0, 64, 64);
                drawpreview(RenderCS.WireframeCPUF64, 1, 128, 128);
                drawpreview(RenderCS.RasterCPUF64, 2, 128, 128);
            }
            if (RenderMode != null)
            {
                drawingContext.DrawText(new FormattedText(RenderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.LightGray), new Point(10, 10));
                drawingContext.DrawText(new FormattedText(RenderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.DarkGray), new Point(8, 8));
            }
        }
        bool renderAgain = false;
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
            Calls = RenderCall.Generate(new[] { typeof(RenderCS), typeof(RenderD3D), typeof(RenderToyCLI) }).ToArray();
            Commands = Calls.ToDictionary(x => x, y => new RoutedUICommand(RenderCall.GetDisplayNameFull(y.MethodInfo.Name), y.MethodInfo.Name, typeof(RenderCallCommands)));
        }
        public static readonly RenderCall[] Calls;
        public static readonly Dictionary<RenderCall, RoutedUICommand> Commands;
    }
}