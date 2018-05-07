////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using RenderToy.SceneGraph.Meshes;
using RenderToy.SceneGraph.Primitives;
using System;
using System.Linq;

namespace RenderToy
{
    public static partial class RenderModeCS
    {
        public static void RasterCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            float[] depthbuffer = new float[render_width * render_height];
            for (int i = 0; i < depthbuffer.Length; ++i) depthbuffer[i] = 1;
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform * mvp;
                uint color = DrawHelp.ColorToUInt32(transformedobject.Node.WireColor);
                // Fill one scanline.
                Action<int, Vector3D, Vector3D> fillscan = (y, x1, x2) =>
                {
                    if (y < 0 || y >= render_height) return;
                    int sx1 = (int)Math.Max(0, Math.Min(render_width, x1.X));
                    int sx2 = (int)Math.Max(0, Math.Min(render_width, x2.X));
                    unsafe
                    {
                        byte* pRaster = (byte*)bitmap_ptr + bitmap_stride * y;
                        for (int scanx = sx1; scanx < sx2; ++scanx)
                        {
                            // Compute pixel depth.
                            float depth = (float)(x1.Z + (x2.Z - x1.Z) * (scanx - sx1) / (sx2 - sx1));
                            // Reject this pixel if it's further away than what we already have.
                            if (depth > depthbuffer[y * render_width + scanx]) continue;
                            // Otherwise set the depth and render the pixel.
                            depthbuffer[y * render_width + scanx] = depth;
                            byte* pPixel = pRaster + 4 * scanx;
                            *(uint*)pPixel = color;
                        }
                    }
                };
                // Fill a triangle defined by 3 points.
                Action<Vector3D, Vector3D, Vector3D> filltri_viewspace = (p1, p2, p3) =>
                {
                    double ymin = Math.Min(p1.Y, Math.Min(p2.Y, p3.Y));
                    double ymax = Math.Max(p1.Y, Math.Max(p2.Y, p3.Y));
                    int yscanmin = Math.Max(0, (int)ymin);
                    int yscanmax = Math.Min(render_height, (int)ymax);
                    // Calculate edge lines.
                    var e1 = new { O = p1, D = p2 - p1 };
                    var e2 = new { O = p2, D = p3 - p2 };
                    var e3 = new { O = p3, D = p1 - p3 };
                    var edges = new[] { e1, e2, e3 };
                    Vector3D o1 = p1, o2 = p2, o3 = p3;
                    Vector3D d1 = p2 - p1, d2 = p3 - p2, d3 = p1 - p3;
                    // Scan in the range of the triangle.
                    for (int y = yscanmin; y <= yscanmax; ++y)
                    {
                        double sy = y;
                        var allx = edges
                            .Select(e => new { E = e, l = (sy - e.O.Y) / e.D.Y })
                            .Where(e => e.l >= 0 && e.l <= 1)
                            .Select(e => e.E.O + e.l * e.E.D)
                            .OrderBy(x => x.X)
                            .ToArray();
                        if (allx.Length == 0) continue;
                        fillscan(y, allx.First(), allx.Last());
                    }
                };
                Action<Vector4D, Vector4D, Vector4D> filltri_clipspace = (p1, p2, p3) =>
                {
                    var iter = ClipHelp.ClipTriangle4D(new Vector4D[] { p1, p2, p3 }).GetEnumerator();
                    while (iter.MoveNext())
                    {
                        var P0 = iter.Current;
                        if (!iter.MoveNext())
                        {
                            break;
                        }
                        var P1 = iter.Current;
                        if (!iter.MoveNext())
                        {
                            break;
                        }
                        var P2 = iter.Current;
                        Vector4D[] v3 = { P0, P1, P2 };
                        Vector3D[] v3t = v3
                            .Select(p => new Vector3D(p.X / p.W, p.Y / p.W, p.Z / p.W))
                            .Select(p => new Vector3D((1 + p.X) * render_width / 2, (1 - p.Y) * render_height / 2, p.Z))
                            .ToArray();
                        filltri_viewspace(v3t[0], v3t[1], v3t[2]);
                    }
                };
                Func<Vector3D, Vector4D> TransformToClip = (p) =>
                {
                    return model_mvp.Transform(new Vector4D(p.X, p.Y, p.Z, 1));
                };
                Action<Vector3D, Vector3D, Vector3D> filltri = (p1, p2, p3) =>
                {
                    filltri_clipspace(TransformToClip(p1), TransformToClip(p2), TransformToClip(p3));
                };
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    for (int v = 0; v < 10; ++v)
                    {
                        for (int u = 0; u < 10; ++u)
                        {
                            Vector3D[] v3 =
                            {
                                uv.GetPointUV((u + 0.0) / 10, (v + 0.0) / 10),
                                uv.GetPointUV((u + 1.0) / 10, (v + 0.0) / 10),
                                uv.GetPointUV((u + 0.0) / 10, (v + 1.0) / 10),
                                uv.GetPointUV((u + 1.0) / 10, (v + 1.0) / 10),
                            };
                            filltri(v3[0], v3[1], v3[3]);
                            filltri(v3[3], v3[2], v3[0]);
                        }
                    }
                    continue;
                }
                Mesh mesh = transformedobject.Node.Primitive as Mesh;
                if (mesh != null)
                {
                    var v = mesh.Vertices;
                    foreach (var t in mesh.Triangles)
                    {
                        filltri(v[t.Index0], v[t.Index1], v[t.Index2]);
                    }
                    continue;
                }
            }
        }
    }
}