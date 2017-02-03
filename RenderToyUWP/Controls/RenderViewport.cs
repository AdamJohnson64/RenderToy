﻿using System;
using System.IO;
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
            flyout.Items.Add(new MenuFlyoutItem { Text = "Point (CPU)", Command = new CommandBinding(o => { RenderMode = RenderModes.Point; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Wireframe (CPU)", Command = new CommandBinding(o => { RenderMode = RenderModes.Wireframe; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raster (CPU)", Command = new CommandBinding(o => { RenderMode = RenderModes.Raster; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast (CPU/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastCPUF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast (CPU/F64)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastCPUF64; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Normals (CPU/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastNormalsCPUF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Normals (CPU/F64)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastNormalsCPUF64; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Tangents (CPU/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastTangentsCPUF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Tangents (CPU/F64)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastTangentsCPUF64; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Bitangents (CPU/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastBitangentsCPUF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Bitangents (CPU/F64)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastBitangentsCPUF64; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raytrace (CPU/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaytraceCPUF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raytrace (CPU/F64)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaytraceCPUF64; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast (AMP/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastAMPF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Normals (AMP/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastNormalsAMPF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Tangents (AMP/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastTangentsAMPF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raycast Bitangents (AMP/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaycastBitangentsAMPF32; }, o => true) });
            flyout.Items.Add(new MenuFlyoutItem { Text = "Raytrace (AMP/F32)", Command = new CommandBinding(o => { RenderMode = RenderModes.RaytraceAMPF32; }, o => true) });
            ContextFlyout = flyout;
            ReduceQuality_Init();
        }
        Image control_image = new Image();
        DispatcherTimer timer = new DispatcherTimer();
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
                case RenderModes.RaycastCPUF32:
                    RenderToyCX.RaycastCPUF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastCPUF64:
                    RenderToyCX.RaycastCPUF64(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastNormalsCPUF32:
                    RenderToyCX.RaycastNormalsCPUF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastNormalsCPUF64:
                    RenderToyCX.RaycastNormalsCPUF64(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastTangentsCPUF32:
                    RenderToyCX.RaycastTangentsCPUF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastTangentsCPUF64:
                    RenderToyCX.RaycastTangentsCPUF64(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastBitangentsCPUF32:
                    RenderToyCX.RaycastBitangentsCPUF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastBitangentsCPUF64:
                    RenderToyCX.RaycastBitangentsCPUF64(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaytraceCPUF32:
                    RenderToyCX.RaytraceCPUF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaytraceCPUF64:
                    RenderToyCX.RaytraceCPUF64(SceneFormatter.CreateFlatMemoryF64(Scene.Default), SceneFormatter.CreateFlatMemoryF64(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastAMPF32:
                    RenderToyCX.RaycastAMPF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastNormalsAMPF32:
                    RenderToyCX.RaycastNormalsAMPF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastTangentsAMPF32:
                    RenderToyCX.RaycastTangentsAMPF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaycastBitangentsAMPF32:
                    RenderToyCX.RaycastBitangentsAMPF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
                case RenderModes.RaytraceAMPF32:
                    RenderToyCX.RaytraceAMPF32(SceneFormatter.CreateFlatMemoryF32(Scene.Default), SceneFormatter.CreateFlatMemoryF32(inverse_mvp), buffer_image, RENDER_WIDTH, RENDER_HEIGHT, 4 * RENDER_WIDTH);
                    break;
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
        enum RenderModes { Point, Wireframe, Raster, RaycastCPUF32, RaycastCPUF64, RaycastNormalsCPUF32, RaycastNormalsCPUF64, RaycastTangentsCPUF32, RaycastTangentsCPUF64, RaycastBitangentsCPUF32, RaycastBitangentsCPUF64, RaytraceCPUF32, RaytraceCPUF64, RaycastAMPF32, RaycastNormalsAMPF32, RaycastTangentsAMPF32, RaycastBitangentsAMPF32, RaytraceAMPF32 }
        RenderModes RenderMode
        {
            get { return renderMode; }
            set { renderMode = value; Repaint(); }
        }
        RenderModes renderMode = RenderModes.RaycastAMPF32;
        #endregion
        #region - Section : Quality Control -
        protected bool ReduceQuality
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
            if (timeEnd.Subtract(timeStart).Milliseconds > 1000 / 30)
            {
                ++reduceQualityFrames;
                if (reduceQualityFrames >= 2 && !reduceQuality)
                {
                    reduceQualityFrames = 0;
                    reduceQuality = true;
                }
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
