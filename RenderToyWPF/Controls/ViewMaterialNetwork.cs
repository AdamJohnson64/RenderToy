////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph.Materials;
using System.Collections.Generic;
using System.Globalization;
using System.Windows;
using System.Windows.Media;

namespace RenderToy.WPF
{
    class ViewMaterialNetwork : FrameworkElement
    {
        public ViewMaterialNetwork()
        {
            ClipToBounds = true;
            Network = MNBrick.Create();
        }
        IMNNode<Vector4D> Network
        {
            set
            {
                network = value;
                InvalidateVisual();
            }
        }
        IMNNode<Vector4D> network;
        protected override void OnRender(DrawingContext drawingContext)
        {
            const int bitmapWidth = 64;
            const int bitmapHeight = 64;
            var typeface = new Typeface("Arial");
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            int span = 0;
            foreach (var node in EnumerateNodes(network))
            {
                System.Type type = node.GetType();
                if (typeof(IMNNode<double>).IsAssignableFrom(type))
                {
                    var cast = (IMNNode<double>)node;
                    if (cast.IsConstant())
                    {
                        var formattedtext = new FormattedText(type.Name + " = " + cast.Eval(new EvalContext()).ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 10, Brushes.Black);
                        drawingContext.DrawText(formattedtext, new Point(0, span));
                        span += 10;
                    }
                    else
                    {
                        var formattedtext = new FormattedText(type.Name, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 10, Brushes.Black);
                        drawingContext.DrawText(formattedtext, new Point(0, span));
                        span += 10;
                        var bitmap = ViewMaterialThumbnail.CreateIMNodeThumbnailDouble(cast, 128, 128);
                        drawingContext.DrawImage(bitmap, new Rect(0, span, bitmapWidth, bitmapHeight));
                        span += bitmapHeight;
                    }
                }
                if (typeof(IMNNode<Vector4D>).IsAssignableFrom(type))
                {
                    var cast = (IMNNode<Vector4D>)node;
                    var bitmap = ViewMaterialThumbnail.CreateIMNodeThumbnailV4D(cast, 128, 128);
                    drawingContext.DrawImage(bitmap, new Rect(0, 0, bitmapWidth, bitmapHeight));
                    span += bitmapHeight;
                }
                var properties = type.GetProperties();
                for (int i = 0; i < properties.Length; ++i)
                {
                    var text = properties[i].Name + " = " + properties[i].GetValue(node);
                    var formattedtext = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 10, Brushes.Black);
                    drawingContext.DrawText(formattedtext, new Point(0, span));
                    span += 10;
                }
            }
        }
        public static IEnumerable<IMNNode> EnumerateNodes(IMNNode node)
        {
            yield return node;
            System.Type type = node.GetType();
            foreach (var property in type.GetProperties())
            {
                if (typeof(IMNNode).IsAssignableFrom(property.PropertyType))
                {
                    foreach (var next in EnumerateNodes((IMNNode)property.GetValue(node)))
                    {
                        yield return next;
                    }
                }
            }
        }
    }
}
