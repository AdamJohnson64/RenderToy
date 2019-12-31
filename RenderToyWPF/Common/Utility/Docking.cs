using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace RenderToy.WPF
{
    interface IToolWindowCreator
    {
        void CreateToolWindow(object content);
    }
    public static class DockTarget
    {
        public static DependencyProperty DockTargetProperty = DependencyProperty.RegisterAttached("DockTarget", typeof(string), typeof(ItemsControl));
        public static string GetDockTarget(ItemsControl on)
        {
            return (string)on.GetValue(DockTargetProperty);
        }
        public static void SetDockTarget(ItemsControl on, string value)
        {
            on.SetValue(DockTargetProperty, value);
        }
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
            var container = parameter as FrameworkElement;
            if (container == null) return;
            var olditemscontainer = ItemsControl.ItemsControlFromItemContainer(container) as ItemsControl;
            if (olditemscontainer == null) return;
            var toolcreator = EnumerateVisualChain(container).OfType<IToolWindowCreator>().FirstOrDefault();
            if (toolcreator == null) return;
            olditemscontainer.Items.Remove(parameter);
            toolcreator.CreateToolWindow(container);
        }
        static IEnumerable<FrameworkElement> EnumerateVisualChain(FrameworkElement element)
        {
            while (element != null)
            {
                yield return element;
                if (element.TemplatedParent == null)
                {
                    element = element.Parent as FrameworkElement;
                }
                else
                {
                    element = element.TemplatedParent as FrameworkElement;
                }
            }
        }
    }
}