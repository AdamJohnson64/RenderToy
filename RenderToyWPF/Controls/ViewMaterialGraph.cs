////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph.Materials;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Media;

namespace RenderToy.WPF
{
    class ViewMaterialGraph : FrameworkElement
    {
        public static DependencyProperty RootProperty = DependencyProperty.Register("Root", typeof(IMNNode), typeof(ViewMaterialGraph), new FrameworkPropertyMetadata(null, OnRootChanged));
        public IMNNode Root
        {
            get { return (IMNNode)GetValue(RootProperty); }
            set { SetValue(RootProperty, value); }
        }
        static void OnRootChanged(object s, DependencyPropertyChangedEventArgs e)
        {
            ((ViewMaterialGraph)s).InvalidateGraph();
        }
        #region - Section : Construction -
        public ViewMaterialGraph()
        {
            ClipToBounds = true;
        }
        #endregion
        #region - Section : Graph Handling -
        class NodePosition
        {
            public double X, Y;
            public IMNNode Node;
            public NodePosition[] Children;
        }
        struct NodeTree
        {
            public IMNNode Node;
            public NodeTree[] Children;
        }
        void InvalidateGraph()
        {
            var root = EnumerateNodeTree(Root);
            var noderoot = CalculateNodePositions(root, 0, 0);
            drawnodes = EnumerateNodePositions(noderoot).ToArray();
            InvalidateVisual();
        }
        static IEnumerable<NodePosition> EnumerateNodePositions(NodePosition root)
        {
            yield return root;
            foreach (var child in root.Children)
            {
                foreach (var next in EnumerateNodePositions(child))
                {
                    yield return next;
                }
            }
        }
        static NodePosition CalculateNodePositions(NodeTree root, double x, double y)
        {
            NodePosition output = new NodePosition();
            output.X = x;
            output.Y = y;
            output.Node = root.Node;
            int childsum = 0;
            int childtotal = CalculateNodeSpan(root);
            List<NodePosition> children = new List<NodePosition>();
            foreach (var child in root.Children)
            {
                var childnext = CalculateNodeSpan(child);
                children.Add(CalculateNodePositions(child, x + 128, y + (childsum + childsum + childnext - childtotal) * 8 / 2));
                childsum += childnext;
            }
            output.Children = children.ToArray();
            return output;
        }
        static int CalculateNodeSpan(NodeTree root)
        {
            if (root.Children.Length == 0) return 1;
            return root.Children.Sum(i => CalculateNodeSpan(i));
        }
        static NodeTree EnumerateNodeTree(IMNNode node)
        {
            NodeTree output;
            output.Node = node;
            var subnodes =
                node.GetType().GetProperties()
                .Where(i => typeof(IMNNode).IsAssignableFrom(i.PropertyType))
                .Select(i => (IMNNode)i.GetValue(node))
                .Distinct()
                .Select(i => EnumerateNodeTree(i));
            output.Children = subnodes.ToArray();
            return output;
        }
        NodePosition[] drawnodes;
        #endregion
        #region - Section : UIElement Overrides -
        protected override void OnRender(DrawingContext drawingContext)
        {
            foreach (var parent in drawnodes)
            {
                foreach (var child in parent.Children)
                {
                    drawingContext.DrawLine(new Pen(Brushes.Black, 1), new Point(parent.X + 100, parent.Y + ActualHeight / 2), new Point(child.X, child.Y + ActualHeight / 2));
                }
            }
            foreach (var node in drawnodes)
            {
                drawingContext.DrawRectangle(Brushes.Red, null, new Rect(node.X, node.Y + ActualHeight / 2 - 3, 100, 6));
            }
        }
        #endregion
    }
}
