////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;

namespace RenderToy.WPF
{
    public static class DockTarget
    {
        public static DependencyProperty DockTargetProperty = DependencyProperty.RegisterAttached("DockTarget", typeof(TabControl), typeof(MainWindow));
        public static bool GetDockTarget(TabControl on)
        {
            return Registered.Contains(on);
        }
        public static void SetDockTarget(TabControl on, bool value)
        {
            if (value)
            {
                Registered.Add(on);
            }
            else
            {
                Registered.Remove(on);
            }
        }
        static HashSet<TabControl> Registered = new HashSet<TabControl>();
    }
    public class CommandLayoutUndockTab : ICommand
    {
        public static ICommand Command = new CommandLayoutUndockTab();
        static CommandLayoutUndockTab()
        {
            CommandManager.RegisterClassCommandBinding(typeof(CommandLayoutUndockTab), new CommandBinding(Command));
        }
        public event EventHandler CanExecuteChanged;
        public bool CanExecute(object parameter)
        {
            return true;
        }
        public void Execute(object parameter)
        {
            var tabitem = parameter as TabItem;
            if (tabitem == null) return;
            var oldtabcontrol = tabitem.Parent as TabControl;
            if (oldtabcontrol == null) return;
            FrameworkElement findhostpresenter = FindPredecessor<ContentPresenter>(oldtabcontrol);
            FrameworkElement findhostwindow = FindPredecessor<Window>(oldtabcontrol);
            var oldhostwindow = findhostpresenter != null ? findhostpresenter : findhostwindow;
            var newtabcontrol = new TabControl();
            oldtabcontrol.Items.Remove(tabitem);
            newtabcontrol.Items.Add(tabitem);
            var window = new Window { Content = newtabcontrol, Title = "Tool Window", Width = 256, Height = 256 };
            window.SetBinding(FrameworkElement.DataContextProperty, new Binding { Source = oldhostwindow, Path = new PropertyPath(FrameworkElement.DataContextProperty) });
            window.Show();
        }
        static T FindPredecessor<T>(FrameworkElement element)
            where T : FrameworkElement
        {
            while (element != null && !(element is T))
            {
                if (element.TemplatedParent == null)
                {
                    element = element.Parent as FrameworkElement;
                }
                else
                {
                    element = element.TemplatedParent as FrameworkElement;
                }
            }
            return (T)element;
        }
    }
}