////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy
{
    class ViewUser : ViewBase
    {
        static RoutedUICommand CommandResolution100 = new RoutedUICommand("100% Resolution", "CommandResolution100", typeof(ViewUser));
        static RoutedUICommand CommandResolution50 = new RoutedUICommand("50% Resolution", "CommandResolution50", typeof(ViewUser));
        static RoutedUICommand CommandResolution25 = new RoutedUICommand("25% Resolution", "CommandResolution25", typeof(ViewUser));
        static RoutedUICommand CommandResolution10 = new RoutedUICommand("10% Resolution", "CommandResolution10", typeof(ViewUser));
        public ViewUser()
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
            var menu = new ContextMenu();
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
        }
        #region - Section : RenderMode Option -
        RenderCall RenderCall
        {
            set
            {
                if (!value.IsMultipass)
                {
                    renderMode = new SinglePassAsyncAdaptor(value, () => Dispatcher.Invoke(InvalidateVisual));
                }
                else
                {
                    renderMode = new MultiPassAsyncAdaptor(value, () => Dispatcher.Invoke(InvalidateVisual));
                }
                renderMode.SetScene(Scene);
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
        protected override void OnSceneChanged(Scene scene)
        {
            base.OnSceneChanged(scene);
            if (RenderMode == null) return;
            RenderMode.SetScene(scene);
        }
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            if (RenderMode == null) return;
            int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth) / RenderResolution;
            int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight) / RenderResolution;
            RenderMode.SetCamera(MVP);
            RenderMode.SetTarget(RENDER_WIDTH, RENDER_HEIGHT);
            if (RENDER_WIDTH == 0 || RENDER_HEIGHT == 0) return;
            WriteableBitmap bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            RenderMode.CopyTo(bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, RENDER_WIDTH, RENDER_HEIGHT));
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
            Calls = RenderCall.Generate(new[] { typeof(RenderCS), typeof(RenderD3D), typeof(RenderToyCLI) }).ToArray();
            Commands = Calls.ToDictionary(x => x, y => new RoutedUICommand(RenderCall.GetDisplayNameFull(y.MethodInfo.Name), y.MethodInfo.Name, typeof(RenderCallCommands)));
        }
        public static readonly RenderCall[] Calls;
        public static readonly Dictionary<RenderCall, RoutedUICommand> Commands;
    }
}