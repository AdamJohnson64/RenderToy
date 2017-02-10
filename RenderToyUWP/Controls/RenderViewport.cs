////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.System;
using Windows.UI.Core;
using Windows.UI.Input;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media.Imaging;

namespace RenderToy
{
    public sealed class RenderViewport : UserControl
    {
        public RenderViewport()
        {
            SizeChanged += (s, e) => { Repaint(); };
            Content = control_image;
            var flyout = new MenuFlyout();
            var calls = RenderCall.Generate(new[] { typeof(RenderCS), typeof(RenderToyCX) }).ToArray();
            foreach (var group in calls.GroupBy(x => RenderCall.GetDisplayNameBare(x.MethodInfo.Name)))
            {
                var flyout_group = new MenuFlyoutSubItem { Text = group.Key };
                foreach (var call in group)
                {
                    flyout_group.Items.Add(new MenuFlyoutItem { Text = RenderCall.GetDisplayNameFull(call.MethodInfo.Name), Command = new CommandBinding(o => { RenderMode = call; }, o => true) });
                }
                flyout.Items.Add(flyout_group);
            }
            renderMode = calls[0];
            ContextFlyout = flyout;
            ReduceQuality_Init();
        }
        Image control_image = new Image();
        #region - Section : Camera -
        TransformPosQuat Camera = new TransformPosQuat { Position = new Point3D(0, 2, -5) };
        #endregion
        #region - Section : Input Handling -
        protected override void OnPointerPressed(PointerRoutedEventArgs e)
        {
            base.OnPointerPressed(e);
            CapturePointer(e.Pointer);
            isDragging = true;
            dragFrom = e.GetCurrentPoint(this);
        }
        protected override void OnPointerReleased(PointerRoutedEventArgs e)
        {
            base.OnPointerReleased(e);
            isDragging = false;
            ReleasePointerCapture(e.Pointer);
        }
        protected override void OnPointerMoved(PointerRoutedEventArgs e)
        {
            base.OnPointerMoved(e);
            if (!isDragging) return;
            PointerPoint dragTo = e.GetCurrentPoint(this);
            double dx = dragTo.Position.X - dragFrom.Position.X;
            double dy = dragTo.Position.Y - dragFrom.Position.Y;
            dragFrom = dragTo;
            // Detect modifier keys.
            var stateLeftControl = CoreWindow.GetForCurrentThread().GetKeyState(VirtualKey.LeftControl);
            var stateLeftShift = CoreWindow.GetForCurrentThread().GetKeyState(VirtualKey.LeftShift);
            bool isPressedLeftControl = (stateLeftControl & CoreVirtualKeyStates.Down) == CoreVirtualKeyStates.Down;
            bool isPressedLeftShift = (stateLeftShift & CoreVirtualKeyStates.Down) == CoreVirtualKeyStates.Down;
            // Process camera motion with modifier keys.
            if (isPressedLeftShift && isPressedLeftControl)
            {
                // Truck Mode (CTRL + SHIFT).
                Camera.TranslatePost(new Vector3D(0, 0, dy * -0.05));
            }
            else if (!isPressedLeftShift && isPressedLeftControl)
            {
                // Rotate Mode (CTRL Only)
                Camera.RotatePre(new Quaternion(new Vector3D(0, 1, 0), dx * 0.05));
                Camera.RotatePost(new Quaternion(new Vector3D(1, 0, 0), dy * 0.05));
            }
            else if (!isPressedLeftShift && !isPressedLeftControl)
            {
                // Translation Mode (no modifier keys).
                Camera.TranslatePost(new Vector3D(dx * -0.05, dy * 0.05, 0));
            }
            Repaint();
        }
        bool isDragging = false;
        PointerPoint dragFrom;
        #endregion
        #region - Section : Rendering -
        void Repaint()
        {
            DateTime timeStart = DateTime.Now;
            int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth);
            int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight);
            if (ReduceQuality)
            {
                RENDER_WIDTH /= 8;
                RENDER_HEIGHT /= 8;
            }
            else
            {
                RENDER_WIDTH /= 2;
                RENDER_HEIGHT /= 2;
            }
            if (RENDER_WIDTH < 8 || RENDER_HEIGHT < 8) return;
            var mvp = MathHelp.Invert(Camera.Transform) * CameraPerspective.CreateProjection(0.01, 100.0, 45, 45) * CameraPerspective.AspectCorrectFit(ActualWidth, ActualHeight);
            var inverse_mvp = MathHelp.Invert(mvp);
            var bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT);
            byte[] buffer_image = new byte[4 * RENDER_WIDTH * RENDER_HEIGHT];
            if (renderMode != null)
            {
                renderMode.Action(Scene.Default, mvp, buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
            }
            using (var stream = bitmap.PixelBuffer.AsStream())
            {
                stream.Write(buffer_image, 0, 4 * RENDER_WIDTH * RENDER_HEIGHT);
            }
            bitmap.Invalidate();
            control_image.Source = bitmap;
            DateTime timeEnd = DateTime.Now;
            ReduceQuality_Decide(timeStart, timeEnd);
        }
        enum RenderModes { Point, Wireframe, Raster, RaycastCPUF32, RaycastCPUF64, RaycastNormalsCPUF32, RaycastNormalsCPUF64, RaycastTangentsCPUF32, RaycastTangentsCPUF64, RaycastBitangentsCPUF32, RaycastBitangentsCPUF64, RaytraceCPUF32, RaytraceCPUF64, RaycastAMPF32, RaycastNormalsAMPF32, RaycastTangentsAMPF32, RaycastBitangentsAMPF32, RaytraceAMPF32, AmbientOcclusionCPUF32, AmbientOcclusionCPUF64, AmbientOcclusionAMPF32 }
        RenderCall RenderMode
        {
            get { return renderMode; }
            set { renderMode = value; Repaint(); }
        }
        RenderCall renderMode;
        #endregion
        #region - Section : Quality Control -
        bool ReduceQuality
        {
            get { return reduceQuality; }
        }
        void ReduceQuality_Init()
        {
            reduceQualityTimer = new DispatcherTimer();
            reduceQualityTimer.Interval = TimeSpan.FromMilliseconds(500);
            reduceQualityTimer.Tick += (s, e) =>
            {
                reduceQualityTimer.Stop();
                reduceQualityFrames = 0;
                reduceQuality = false;
                Repaint();
            };
        }
        void ReduceQuality_Decide(DateTime timeStart, DateTime timeEnd)
        {
            // If we're below 30FPS then reduce the quality.
            if (timeEnd.Subtract(timeStart).Milliseconds > 1000 / 30)
            {
                ++reduceQualityFrames;
                if (reduceQualityFrames >= 2 && !reduceQuality)
                {
                    reduceQualityFrames = 0;
                    reduceQuality = true;
                }
            }
            // If we're exceeding 90FPS and in reduced quality then increase the quality.
            if (timeEnd.Subtract(timeStart).Milliseconds < 1000 / 90 && reduceQuality)
            {
                reduceQualityFrames = 0;
                reduceQuality = false;
            }
            // Restart the quality reduction timer.
            if (reduceQuality)
            {
                reduceQualityTimer.Stop();
                reduceQualityTimer.Start();
            }
        }
        bool reduceQuality = false;
        int reduceQualityFrames = 0;
        DispatcherTimer reduceQualityTimer;
        #endregion
    }
    class CommandBinding : System.Windows.Input.ICommand
    {
        public CommandBinding(Action<object> execute, Func<object, bool> canexecute)
        {
            this.canexecute = canexecute;
            this.execute = execute;
        }
        public event EventHandler CanExecuteChanged;
        public bool CanExecute(object parameter)
        {
            return canexecute(parameter);
        }
        public void Execute(object parameter)
        {
            execute(parameter);
        }
        Func<object, bool> canexecute;
        Action<object> execute;
    }
}
