﻿using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.System;
using Windows.UI.Core;
using Windows.UI.Input;
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
            flyout.Items.Add(new MenuFlyoutItem { Text = "Point", Command = new CommandBinding(o => { RenderMode = RenderModes.Point; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Wireframe", Command = new CommandBinding(o => { RenderMode = RenderModes.Wireframe; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raster", Command = new CommandBinding(o => { RenderMode = RenderModes.Raster; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast (CPU)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastCPU; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Normals (CPU)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastNormalsCPU; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Tangents (CPU)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastTangentsCPU; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Bitangents (CPU)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastBitangentsCPU; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raytrace (CPU/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaytraceCPUF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raytrace (CPU/F64)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaytraceCPUF64; }, o => true) });
            ContextFlyout = flyout;
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
            int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth) / 4;
            int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight) / 4;
            if (RENDER_WIDTH < 8 || RENDER_HEIGHT < 8) return;
            var mvp = MathHelp.Invert(Camera.Transform) * CameraPerspective.CreateProjection(0.01, 100.0, 45, 45) * CameraPerspective.AspectCorrectFit(ActualWidth, ActualHeight);
            var inverse_mvp = MathHelp.Invert(mvp);
            var bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT);
            byte[] buffer_image = new byte[4 * RENDER_WIDTH * RENDER_HEIGHT];
            switch (renderMode)
            {
                case RenderModes.Point:
                    {
                        GCHandle gchandle = GCHandle.Alloc(buffer_image, GCHandleType.Pinned);
                        try
                        {
                            RenderCS.Point(Scene.Default, mvp, gchandle.AddrOfPinnedObject(), RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                        }
                        finally
                        {
                            gchandle.Free();
                        }
                    }
                    break;
                case RenderModes.Wireframe:
                    {
                        GCHandle gchandle = GCHandle.Alloc(buffer_image, GCHandleType.Pinned);
                        try
                        {
                            RenderCS.Wireframe(Scene.Default, mvp, gchandle.AddrOfPinnedObject(), RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                        }
                        finally
                        {
                            gchandle.Free();
                        }
                    }
                    break;
                case RenderModes.Raster:
                    {
                        GCHandle gchandle = GCHandle.Alloc(buffer_image, GCHandleType.Pinned);
                        try
                        {
                            RenderCS.Raster(Scene.Default, mvp, gchandle.AddrOfPinnedObject(), RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                        }
                        finally
                        {
                            gchandle.Free();
                        }
                    }
                    break;
                case RenderModes.RaycastCPU:
                    RenderToyCX.RaycastCPU(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastNormalsCPU:
                    RenderToyCX.RaycastNormalsCPU(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastTangentsCPU:
                    RenderToyCX.RaycastTangentsCPU(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastBitangentsCPU:
                    RenderToyCX.RaycastBitangentsCPU(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaytraceCPUF32:
                    RenderToyCX.RaytraceCPUF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaytraceCPUF64:
                    RenderToyCX.RaytraceCPUF64(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
            }
            using (var stream = bitmap.PixelBuffer.AsStream())
            {
                stream.Write(buffer_image, 0, 4 * RENDER_WIDTH * RENDER_HEIGHT);
            }
            bitmap.Invalidate();
            control_image.Source = bitmap;
        }
        enum RenderModes { Point, Wireframe, Raster, RaycastCPU, RaycastNormalsCPU, RaycastTangentsCPU, RaycastBitangentsCPU, RaytraceCPUF32, RaytraceCPUF64 }
        RenderModes RenderMode
        {
            get { return renderMode; }
            set { renderMode = value; Repaint(); }
        }
        RenderModes renderMode = RenderModes.RaytraceCPUF32;
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
