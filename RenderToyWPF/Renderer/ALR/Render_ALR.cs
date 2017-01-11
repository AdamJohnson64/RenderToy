////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static partial class Render
    {
        #region - Section : Phase 2 - Wireframe Rendering (Abstract Line Renderer) -
        private static void DrawWireframeCommon(Scene scene, Matrix3D mvp, IWireframeRenderer renderer, double width, double height)
        {
            DrawHelp.fnDrawLineViewport lineviewport = CreateLineViewportFunction(renderer, width, height);
            renderer.WireframeBegin();
            // Draw something interesting.
            renderer.WireframeColor(0.0, 0.0, 0.0);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv == null) continue;
                DrawHelp.fnDrawLineWorld line = CreateLineWorldFunction(lineviewport, transformedobject.Transform * mvp);
                Color color = transformedobject.Node.WireColor;
                renderer.WireframeColor(color.R / 255.0 / 2, color.G / 255.0 / 2, color.B / 255.0 / 2);
                DrawHelp.DrawParametricUV(line, uv);
            }
            renderer.WireframeEnd();
        }
        public static DrawHelp.fnDrawLineViewport CreateLineViewportFunction(IWireframeRenderer renderer, double width, double height)
        {
            return (p1, p2) =>
            {
                renderer.WireframeLine(
                    (p1.X + 1) * width / 2, (1 - p1.Y) * height / 2,
                    (p2.X + 1) * width / 2, (1 - p2.Y) * height / 2);
            };
        }
        public static DrawHelp.fnDrawLineWorld CreateLineWorldFunction(DrawHelp.fnDrawLineViewport line, Matrix3D mvp)
        {
            return (p1, p2) =>
            {
                DrawHelp.DrawLineWorld(line, mvp, new Point4D(p1.X, p1.Y, p1.Z, 1.0), new Point4D(p2.X, p2.Y, p2.Z, 1.0));
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