﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace RenderToy.WPF
{
    class ViewMaterialGraph : FrameworkElement
    {
        #region - Section : Dependency Properties -
        public static DependencyProperty RootProperty = DependencyProperty.Register("Root", typeof(IMNNode), typeof(ViewMaterialGraph), new FrameworkPropertyMetadata(null, OnRootChanged));
        public IMNNode Root
        {
            get { return (IMNNode)GetValue(RootProperty); }
            set { SetValue(RootProperty, value); }
        }
        public static DependencyProperty NodeTemplateProperty = DependencyProperty.Register("NodeTemplate", typeof(DataTemplate), typeof(ViewMaterialGraph), new FrameworkPropertyMetadata(null, OnRootChanged));
        public DataTemplate NodeTemplate
        {
            get { return (DataTemplate)GetValue(NodeTemplateProperty); }
            set { SetValue(NodeTemplateProperty, value); }
        }
        static void OnRootChanged(object s, DependencyPropertyChangedEventArgs e)
        {
            ((ViewMaterialGraph)s).InvalidateGraph();
        }
        #endregion
        #region - Section : Construction -
        public ViewMaterialGraph()
        {
            ClipToBounds = true;
        }
        #endregion
        #region - Section : Graph Handling -
        class NodeConnection
        {
            public PropertyInfo Origin;
            public NodePosition Target;
        }
        class NodePosition
        {
            public double X, Y;
            public IMNNode Node;
            public Visual Visual;
            public NodeConnection[] Children;
        }
        struct NodeTree
        {
            public IMNNode Node;
            public NodeTree[] Children;
        }
        void InvalidateGraph()
        {
            if (visuals != null)
            {
                foreach (var visual in visuals)
                {
                    RemoveVisualChild(visual.Value);
                }
            }
            visuals = new Dictionary<NodePosition, Visual>();
            visualroot = GenerateVisualTree(Root);
            LayoutGraph();
        }
        void LayoutGraph()
        {
            UpdateLayout();
            LayoutGraph(visualroot, 0, ActualHeight / 2);
            InvalidateArrange();
            InvalidateVisual();
        }
        static void LayoutGraph(NodePosition node, double x, double y)
        {
            node.X = x;
            node.Y = y;
            double mywidth = GetWidth(node);
            double myheight = CalculateHeight(node);
            double minrange = y - myheight / 2;
            foreach (var child in node.Children)
            {
                double maxrange = minrange + CalculateHeight(child.Target);
                LayoutGraph(child.Target, x + mywidth + 32, (minrange + maxrange) / 2);
                minrange = maxrange;
            }
        }
        static double GetWidth(NodePosition node)
        {
            var ui = node.Visual as UIElement;
            if (ui == null) return 0;
            return ui.DesiredSize.Width;
        }
        static double GetHeight(NodePosition node)
        {
            var ui = node.Visual as UIElement;
            if (ui == null) return 0;
            return ui.DesiredSize.Height;
        }
        static double CalculateHeight(NodePosition node)
        {
            var ui = node.Visual as UIElement;
            if (ui == null) return 0;
            double myheight = ui.DesiredSize.Height;
            double nextheight = node.Children.Sum(i => CalculateHeight(i.Target));
            return Math.Max(myheight, nextheight);
        }
        NodePosition GenerateVisualTree(IMNNode node)
        {
            NodePosition output = new NodePosition();
            output.Node = node;
            var subnodes =
                node.GetType().GetProperties()
                .Where(i => typeof(IMNNode).IsAssignableFrom(i.PropertyType))
                .Select(i => new { Origin = i, Value = (IMNNode)i.GetValue(node) })
                .Select(i => new NodeConnection { Origin = i.Origin, Target = GenerateVisualTree(i.Value) });
            output.Children = subnodes.ToArray();
            if (NodeTemplate != null)
            {
                var obj = NodeTemplate.LoadContent();
                var visual = obj as Visual;
                if (visual != null)
                {
                    output.Visual = visual;
                    visuals.Add(output, output.Visual);
                    AddVisualChild(output.Visual);
                    var frameworkelement = output.Visual as FrameworkElement;
                    if (frameworkelement != null)
                    {
                        frameworkelement.DataContext = node;
                    }
                    var ui = output.Visual as UIElement;
                    if (ui != null)
                    {
                        ui.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
                    }
                }
            }
            return output;
        }
        Dictionary<NodePosition, Visual> visuals = new Dictionary<NodePosition, Visual>();
        NodePosition visualroot;
        #endregion
        #region - Section : Overrides -
        protected override int VisualChildrenCount => visuals.Count;
        protected override Visual GetVisualChild(int index)
        {
            return visuals.Values.ElementAt(index);
        }
        protected override Size ArrangeOverride(Size arrangeBounds)
        {
            int visualcount = visuals.Count;
            foreach (var node in visuals)
            {
                var visual = node.Value as UIElement;
                if (visual == null) continue;
                visual.Arrange(new Rect(node.Key.X, node.Key.Y, visual.DesiredSize.Width, visual.DesiredSize.Height));
            }
            return arrangeBounds;
        }
        protected override Size MeasureOverride(Size constraint)
        {
            int visualcount = visuals.Count;
            for (int i = 0; i < visualcount; ++i)
            {
                var visual = GetVisualChild(i) as UIElement;
                if (visual == null) continue;
                visual.Measure(constraint);
            }
            return constraint;
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            foreach (var parent in visuals.Keys)
            {
                var interfaces = parent.Visual as INodeInputHandle;
                if (interfaces == null) continue;
                foreach (var child in parent.Children)
                {
                    var interfacepoint = interfaces.GetInputHandleLocation(child.Origin);
                    drawingContext.DrawLine(new Pen(Brushes.Black, 1), new Point(parent.X + interfacepoint.X, parent.Y + interfacepoint.Y), new Point(child.Target.X, child.Target.Y + GetHeight(child.Target) / 2));
                }
            }
        }
        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            base.OnMouseDown(e);
            LayoutGraph();
        }
        #endregion
    }
}
