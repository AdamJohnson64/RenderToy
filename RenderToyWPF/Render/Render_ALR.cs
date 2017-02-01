////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Windows.Media;

namespace RenderToy
{
    public static partial class AbstractLineRenderer
    {
        #region - Section : Phase 2 - Wireframe Rendering (Abstract Line Renderer) -
        public static void DrawWireframe(Scene scene, Matrix3D mvp, IWireframeRenderer renderer, double width, double height)
        {
            renderer.WireframeBegin();
            // Draw something interesting.
            renderer.WireframeColor(0.0, 0.0, 0.0);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv == null) continue;
                DrawHelp.fnDrawLineWorld line = CreateLineWorldFunction(renderer, width, height, transformedobject.Transform * mvp);
                Color color = transformedobject.Node.WireColor;
                renderer.WireframeColor(color.R / 255.0 / 2, color.G / 255.0 / 2, color.B / 255.0 / 2);
                DrawHelp.DrawParametricUV(line, uv.GetPointUV);
            }
            renderer.WireframeEnd();
        }
        public static DrawHelp.fnDrawLineWorld CreateLineWorldFunction(IWireframeRenderer renderer, double width, double height, Matrix3D mvp)
        {
            return (p1, p2) =>
            {
                Point4D v41 = new Point4D(p1.X, p1.Y, p1.Z, 1);
                Point4D v42 = new Point4D(p2.X, p2.Y, p2.Z, 1);
                if (!ClipHelp.TransformAndClipLine(ref v41, ref v42, mvp)) return;
                // Perform homogeneous divide and draw the viewport space line.
                renderer.WireframeLine(
                    (v41.X / v41.W + 1) * width / 2, (1 - v41.Y / v41.W) * height / 2,
                    (v42.X / v42.W + 1) * width / 2, (1 - v42.Y / v42.W) * height / 2);
            };
        }
        #endregion
    }
    public interface IWireframeRenderer
    {
        void WireframeBegin();
        void WireframeColor(double r, double g, double b);
        void WireframeLine(double x1, double y1, double x2, double y2);
        void WireframeEnd();
    }
}