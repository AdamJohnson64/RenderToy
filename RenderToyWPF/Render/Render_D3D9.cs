////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Linq;

namespace RenderToy
{
    public static class RenderD3D
    {
        #region - Section : Phase 3 - Rasterized Rendering (Direct3D 9) -
        public static void RasterD3D9(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            D3D9Surface d3dsurface = new D3D9Surface(render_width, render_height);
            d3dsurface.BeginScene();
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform * mvp;
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv == null) continue;
                d3dsurface.SetColor(DrawHelp.ColorToUInt32(transformedobject.Node.WireColor));
                Action<Point4D, Point4D, Point4D> filltri_clipspace = (p1, p2, p3) =>
                {
                    foreach (var tri in ClipHelp.ClipTriangle3D(new ClipHelp.Triangle { p1 = p1, p2 = p2, p3 = p3 }))
                    {
                        Point4D[] points = { tri.p1, tri.p2, tri.p3 };
                        var t = points
                            .Select(p => new Point4D(p.X / p.W, p.Y / p.W, p.Z / p.W, 1))
                            .ToArray();
                        d3dsurface.DrawTriangle(
                            (float)t[0].X, (float)t[0].Y, (float)t[0].Z, (float)t[0].W,
                            (float)t[1].X, (float)t[1].Y, (float)t[1].Z, (float)t[1].W,
                            (float)t[2].X, (float)t[2].Y, (float)t[2].Z, (float)t[2].W);
                    }
                };
                for (int v = 0; v < 10; ++v)
                {
                    for (int u = 0; u < 10; ++u)
                    {
                        Point3D[] points =
                        {
                            uv.GetPointUV((u + 0.0) / 10, (v + 0.0) / 10),
                            uv.GetPointUV((u + 1.0) / 10, (v + 0.0) / 10),
                            uv.GetPointUV((u + 0.0) / 10, (v + 1.0) / 10),
                            uv.GetPointUV((u + 1.0) / 10, (v + 1.0) / 10),
                        };
                        var t = points
                            .Select(x => new Point4D(x.X, x.Y, x.Z, 1))
                            .Select(x => model_mvp.Transform(x))
                            .ToArray();
                        filltri_clipspace(t[0], t[1], t[3]);
                        filltri_clipspace(t[3], t[2], t[0]);
                    }
                }
            }
            d3dsurface.EndScene();
            d3dsurface.CopyTo(bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        #endregion
    }
}
