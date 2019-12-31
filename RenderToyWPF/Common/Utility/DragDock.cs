////////////////////////////////////////////////////////////////////////////////
// We're currently only using this control as a quick and dirty way to master
// a layout and gain insights into their usability. Dragging doesn't work here.
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public enum DragDockPosition
    {
        Center,
        SliceLeft,
        SliceRight,
        SliceTop,
        SliceBottom,
        AddLeft,
        AddRight,
        AddTop,
        AddBottom,
    }
    public class DragDock : Panel
    {
        public static readonly DependencyProperty DockProperty = DependencyProperty.RegisterAttached("Dock", typeof(DragDockPosition), typeof(DragDock));
        public static DragDockPosition GetDragDockPosition(DependencyObject on)
        {
            return (DragDockPosition)on.GetValue(DockProperty);
        }
        public static void SetDragDockPosition(DependencyObject on, DragDockPosition value)
        {
            on.SetValue(DockProperty, value);
        }
        public static readonly DependencyProperty TitleProperty = DependencyProperty.RegisterAttached("Title", typeof(string), typeof(DragDock), new FrameworkPropertyMetadata("Untitled"));
        public static string GetTitle(DependencyObject on)
        {
            return (string)on.GetValue(TitleProperty);
        }
        public static void SetTitle(DependencyObject on, string value)
        {
            on.SetValue(TitleProperty, value);
        }
        protected override void OnRender(DrawingContext dc)
        {
            base.OnRender(dc);
            foreach (var region in AllRegions)
            {
                if (region.Children.Count == 1)
                {
                    dc.DrawRectangle(Brushes.DarkSlateGray, new Pen(new SolidColorBrush(Colors.DarkGray), 1), region.Area);
                    var formattedtext = new FormattedText(GetTitle(region.Children[0]), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.LightGray, 1.0);
                    dc.DrawText(formattedtext, new Point(region.Area.X + 4, region.Area.Y + 4));
                }
                else
                {
                    dc.DrawRectangle(Brushes.DarkSlateGray, new Pen(new SolidColorBrush(Colors.DarkGray), 1), region.Area);
                    double tabOffsetX = 0;
                    foreach (var child in region.Children)
                    {
                        var formattedtext = new FormattedText(GetTitle(child), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 12, Brushes.DarkGray, 1.0);
                        dc.DrawRectangle(Brushes.Black, new Pen(new SolidColorBrush(Colors.DarkGray), 1), new Rect(region.Area.X + 2 + tabOffsetX, region.Area.Y + 2, formattedtext.Width + 4, 16));
                        dc.DrawText(formattedtext, new Point(region.Area.X + 4 + tabOffsetX, region.Area.Y + 4));
                        tabOffsetX += formattedtext.Width + 8;
                    }
                }
            }
        }
        protected override Size ArrangeOverride(Size finalSize)
        {
            foreach (var region in AllRegions)
            {
                Rect outer = region.Area;
                Rect inner = region.Area;
                inner.X += 2;
                inner.Y += 20;
                inner.Width = System.Math.Max(0, inner.Width - 2 - 2);
                inner.Height = System.Math.Max(0, inner.Height - 20 - 2);
                region.Children[0].Arrange(inner);
            }
            return base.ArrangeOverride(finalSize);
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            {
                var allregions = new List<DockRegion>();
                var middle = new DockRegion { Area = new Rect(0, 0, availableSize.Width, availableSize.Height) };
                allregions.Add(middle);
                var sides = new DockRegion[4];
                foreach (UIElement child in Children)
                {
                    var position = GetDragDockPosition(child);
                    switch (position)
                    {
                        case DragDockPosition.Center:
                            middle.Children.Add(child);
                            break;
                        case DragDockPosition.SliceLeft:
                            sides[0] = new DockRegion { Area = new Rect(middle.Area.X, middle.Area.Y, 128, middle.Area.Height) };
                            sides[0].Children.Add(child);
                            allregions.Add(sides[0]);
                            middle.Area.X += 128;
                            middle.Area.Width = System.Math.Max(0, middle.Area.Width - 128);
                            break;
                        case DragDockPosition.SliceRight:
                            sides[1] = new DockRegion { Area = new Rect(middle.Area.Right - 128, middle.Area.Y, 128, middle.Area.Height) };
                            sides[1].Children.Add(child);
                            allregions.Add(sides[1]);
                            middle.Area.Width = System.Math.Max(0, middle.Area.Width - 128);
                            break;
                        case DragDockPosition.SliceTop:
                            sides[2] = new DockRegion { Area = new Rect(middle.Area.X, middle.Area.Y, middle.Area.Width, 128) };
                            sides[2].Children.Add(child);
                            allregions.Add(sides[2]);
                            middle.Area.Y += 128;
                            middle.Area.Height = System.Math.Max(0, middle.Area.Height - 128);
                            break;
                        case DragDockPosition.SliceBottom:
                            sides[3] = new DockRegion { Area = new Rect(middle.Area.X, middle.Area.Bottom - 128, middle.Area.Width, 128) };
                            sides[3].Children.Add(child);
                            allregions.Add(sides[3]);
                            middle.Area.Height = System.Math.Max(0, middle.Area.Height - 128);
                            break;
                        case DragDockPosition.AddLeft:
                            sides[0].Children.Add(child);
                            break;
                        case DragDockPosition.AddRight:
                            sides[1].Children.Add(child);
                            break;
                        case DragDockPosition.AddTop:
                            sides[2].Children.Add(child);
                            break;
                        case DragDockPosition.AddBottom:
                            sides[3].Children.Add(child);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                }
                AllRegions = allregions;
            }
            foreach (var region in AllRegions)
            {
                Rect outer = region.Area;
                Rect inner = region.Area;
                inner.X += 2;
                inner.Y += 20;
                inner.Width = System.Math.Max(0, inner.Width - 2 - 2);
                inner.Height = System.Math.Max(0, inner.Height - 20 - 2);
                region.Children[0].Measure(inner.Size);
                // TODO: Tab switching.
                for (int i = 1; i < region.Children.Count; ++i)
                {
                    region.Children[i].Measure(new Size(0, 0));
                }
            }
            return base.MeasureOverride(availableSize);
        }
        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            base.OnMouseDown(e);
            foreach (UIElement child in Children)
            {
                if (new Rect(new Point(0, 0), child.RenderSize).Contains(e.GetPosition(child)))
                {
                    Debug.WriteLine("Click in '" + child + "'.");
                }
            }
        }
        class DockRegion
        {
            public Rect Area;
            public List<UIElement> Children = new List<UIElement>();
        }
        List<DockRegion> AllRegions = new List<DockRegion>();
    }
}