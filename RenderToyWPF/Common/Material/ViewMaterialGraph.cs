using RenderToy.Materials;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Markup;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public class ViewMaterialGraph : FrameworkElement
    {
        #region - Section : Dependency Properties -
        public static DependencyProperty RootProperty = DependencyProperty.Register("Root", typeof(IMaterial), typeof(ViewMaterialGraph), new FrameworkPropertyMetadata(null, OnRootChanged));
        public IMaterial Root
        {
            get { return (IMaterial)GetValue(RootProperty); }
            set { SetValue(RootProperty, value); }
        }
        public static DependencyProperty NodeTemplateProperty = DependencyProperty.Register("NodeTemplate", typeof(DataTemplate), typeof(ViewMaterialGraph), new FrameworkPropertyMetadata(null, OnRootChanged));
        public DataTemplate NodeTemplate
        {
            get { return (DataTemplate)GetValue(NodeTemplateProperty); }
            set { SetValue(NodeTemplateProperty, value); }
        }
        public static DependencyProperty ForegroundProperty = DependencyProperty.Register("Foreground", typeof(Color), typeof(ViewMaterialGraph), new FrameworkPropertyMetadata(Colors.Black, FrameworkPropertyMetadataOptions.AffectsRender));
        public Color Foreground
        {
            get { return (Color)GetValue(ForegroundProperty); }
            set { SetValue(ForegroundProperty, value); }
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
            public PropertyInfo[] Origin;
            public NodePosition Target;
        }
        class NodePosition
        {
            public double X, Y;
            public IMaterial Node;
            public Visual Visual;
            public NodeConnection[] Children;
        }
        void InvalidateGraph()
        {
            // Disconnect all the previous visuals.
            if (visuals != null)
            {
                foreach (var visual in visuals)
                {
                    RemoveVisualChild(visual.Value);
                }
            }
            // Rebuild the visuals and re-add them.
            visuals = new Dictionary<NodePosition, Visual>();
            visualroot = GenerateVisualTree(Root);
            // Kick the layout engine to position and size the new graph.
            InvalidateArrange();
            // Kick the render engine to redraw connections.
            InvalidateVisual();
        }
        static void LayoutGraph(NodePosition root)
        {
            // Position the root so the topmost span reaches exactly Y=0.
            LayoutGraph(root, 0, CalculateBranchHeight(root) / 2);
        }
        static void LayoutGraph(NodePosition node, double x, double y)
        {
            if (node == null) return;
            node.X = x;
            node.Y = y;
            double mywidth = GetWidth(node);
            double myheight = CalculateBranchHeight(node);
            double minrange = y - myheight / 2;
            foreach (var child in node.Children)
            {
                double maxrange = minrange + CalculateBranchHeight(child.Target);
                LayoutGraph(child.Target, x + mywidth + 32, (minrange + maxrange) / 2);
                minrange = maxrange;
            }
        }
        static double GetWidth(NodePosition node)
        {
            var ui = node.Visual as UIElement;
            return ui == null ? 0 : ui.DesiredSize.Width;
        }
        static double GetHeight(NodePosition node)
        {
            var ui = node.Visual as UIElement;
            return ui == null ? 0 : ui.DesiredSize.Height;
        }
        static double CalculateBranchHeight(NodePosition node)
        {
            if (node == null) return 0;
            var ui = node.Visual as UIElement;
            if (ui == null) return 0;
            double myheight = ui.DesiredSize.Height;
            double nextheight = node.Children.Sum(i => CalculateBranchHeight(i.Target));
            return System.Math.Max(myheight, nextheight);
        }
        NodePosition GenerateVisualTree(IMaterial node)
        {
            if (node == null) return null;
            NodePosition output = new NodePosition();
            output.Node = node;
            var subnodes =
                node.GetType().GetProperties()
                .Where(i => typeof(IMaterial).IsAssignableFrom(i.PropertyType))
                .Select(i => new { Origin = i, Value = (IMaterial)i.GetValue(node) })
                .GroupBy(i => i.Value)
                .Select(i => new NodeConnection { Origin = i.Select(j => j.Origin).ToArray(), Target = GenerateVisualTree(i.Key) });
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
                    var frameworkelement = output.Visual as ContentPresenter;
                    if (frameworkelement != null)
                    {
                        frameworkelement.Content = node;
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
        static IEnumerable<NodePosition> EnumerateNodes(NodePosition root)
        {
            if (root == null) yield break;
            yield return root;
            foreach (var child in root.Children)
            {
                foreach (var next in EnumerateNodes(child.Target))
                {
                    yield return next;
                }
            }
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
            // Pre-measure all children to obtain their desired sizes.
            int visualcount = visuals.Count;
            for (int i = 0; i < visualcount; ++i)
            {
                var visual = GetVisualChild(i) as UIElement;
                if (visual == null) continue;
                // Ignore our actual layout size - we'll position as best we can in virtual space.
                visual.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
            }
            // If we don't have a graph then ignore.
            if (visualroot == null) return new Size(0, 0);
            // Arrange all the interior nodes.
            LayoutGraph(visualroot);
            // Walk the visual tree and fit all the nodes.
            double minx = double.PositiveInfinity;
            double miny = double.PositiveInfinity;
            double maxx = double.NegativeInfinity;
            double maxy = double.NegativeInfinity;
            foreach (var node in EnumerateNodes(visualroot))
            {
                var visual = node.Visual as UIElement;
                if (visual == null) continue;
                minx = System.Math.Min(minx, node.X);
                miny = System.Math.Min(miny, node.Y);
                maxx = System.Math.Max(maxx, node.X + visual.DesiredSize.Width);
                maxy = System.Math.Max(maxy, node.Y + visual.DesiredSize.Height);
            }
            // Request enough size for the whole graph.
            return new Size(System.Math.Max(0, maxx), System.Math.Max(0, maxy));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var pen = new Pen(new SolidColorBrush(Foreground), 1);
            foreach (var parent in visuals.Keys)
            {
                var interfaces = parent.Visual as INodeInputHandle;
                if (interfaces == null) continue;
                foreach (var child in parent.Children)
                {
                    if (child == null || child.Target == null) continue;
                    foreach (var origin in child.Origin)
                    {
                        var interfacepoint = interfaces.GetInputHandleLocation(origin);
                        drawingContext.DrawLine(pen, new Point(parent.X + interfacepoint.X, parent.Y + interfacepoint.Y), new Point(child.Target.X, child.Target.Y + GetHeight(child.Target) / 2));
                    }
                }
            }
        }
        #endregion
    }
    class TypeBasedDataTemplate
    {
        public Type DataType { get; set; }
        public DataTemplate DataTemplate { get; set; }
    }
    [ContentProperty("Templates")]
    class TypeBasedDataTemplateSelector : DataTemplateSelector
    {
        public List<TypeBasedDataTemplate> Templates { get { return templates; } }
        List<TypeBasedDataTemplate> templates = new List<TypeBasedDataTemplate>();
        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            if (item == null) return null;
            var find = templates.FirstOrDefault(i => i.DataType == null || i.DataType.IsAssignableFrom(item.GetType()));
            if (find == null) return null;
            return find.DataTemplate;
        }
    }
}
