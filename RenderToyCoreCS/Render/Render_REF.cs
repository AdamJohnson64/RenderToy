﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Linq;

namespace RenderToy
{
    public static partial class RenderCS
    {
        #region - Section : Phase 1 - Point Rendering (Reference) -
        public static void PointCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform * mvp;
                uint color = DrawHelp.ColorToUInt32(transformedobject.Node.WireColor);
                // Draw a single pixel to the framebuffer (safety function).
                Action<int, int> drawpixel2d = (x, y) =>
                {
                    if (!(x >= 0 && x < render_width && y >= 0 && y < render_height)) return;
                    unsafe
                    {
                        byte* pRaster = (byte*)bitmap_ptr + bitmap_stride * y;
                        byte* pPixel = pRaster + 4 * x;
                        *(uint*)pPixel = color;
                    }
                };
                // Transform, clip and render a 3D point at P.
                DrawHelp.fnDrawPointWorld drawpoint = (p) =>
                {
                    Vector4D v4 = new Vector4D(p.X, p.Y, p.Z, 1);
                    v4 = model_mvp.Transform(v4);
                    if (v4.W <= 0) return;
                    drawpixel2d((int)((1 + v4.X / v4.W) * render_width / 2), (int)((1 - v4.Y / v4.W) * render_height / 2));
                };
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    DrawHelp.DrawParametricUV(drawpoint, uv.GetPointUV);
                    continue;
                }
                IParametricUVW uvw = transformedobject.Node.Primitive as IParametricUVW;
                if (uvw != null)
                {
                    DrawHelp.DrawParametricUVW(drawpoint, uvw.GetPointUVW);
                    continue;
                }
                Mesh mesh = transformedobject.Node.Primitive as Mesh;
                if (mesh != null)
                {
                    foreach (var p in mesh.Vertices)
                    {
                        drawpoint(p);
                    }
                    continue;
                }
                /*
                MeshBVH meshbvh = transformedobject.Node.Primitive as MeshBVH;
                if (meshbvh != null)
                {
                    foreach (var p in meshbvh.Vertices)
                    {
                        drawpoint(p);
                    }
                    continue;
                }
                */
            }
        }
        #endregion
        #region - Section : Phase 2 - Wireframe Rendering (Reference) -
        public static void WireframeCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform * mvp;
                uint color = DrawHelp.ColorToUInt32(transformedobject.Node.WireColor);
                // Draw a single pixel to the framebuffer (safety function).
                Action<int, int> drawpixel2d = (x, y) =>
                {
                    if (!(x >= 0 && x < render_width && y >= 0 && y < render_height)) return;
                    unsafe
                    {
                        byte* pRaster = (byte*)bitmap_ptr + bitmap_stride * y;
                        byte* pPixel = pRaster + 4 * x;
                        *(uint*)pPixel = color;
                    }
                };
                // Draw a line to the framebuffer.
                DrawHelp.fnDrawLineWorld drawline2d = (p1, p2) =>
                {
                    if (Math.Abs(p2.X - p1.X) > Math.Abs(p2.Y - p1.Y))
                    {
                        // X spanning line; this line is longer in the X axis.
                        // Scan in the X direction plotting Y points.
                        if (p1.X > p2.X)
                        {
                            Vector3D t = p1;
                            p1 = p2;
                            p2 = t;
                        }
                        for (int x = 0; x < p2.X - p1.X; ++x)
                        {
                            drawpixel2d((int)(p1.X + x), (int)(p1.Y + (p2.Y - p1.Y) * x / (p2.X - p1.X)));
                        }
                    }
                    else
                    {
                        // Y spanning line; this line is longer in the Y axis.
                        // Scan in the Y direction plotting X points.
                        if (p1.Y > p2.Y)
                        {
                            Vector3D t = p1;
                            p1 = p2;
                            p2 = t;
                        }
                        for (int y = 0; y < p2.Y - p1.Y; ++y)
                        {
                            drawpixel2d((int)(p1.X + (p2.X - p1.X) * y / (p2.Y - p1.Y)), (int)(p1.Y + y));
                        }
                    }
                };
                // Transform, clip and render a 3D line between P1 and P2.
                DrawHelp.fnDrawLineWorld drawline3d = (p1, p2) => {
                    Vector4D v41 = new Vector4D(p1.X, p1.Y, p1.Z, 1);
                    Vector4D v42 = new Vector4D(p2.X, p2.Y, p2.Z, 1);
                    if (!ClipHelp.TransformAndClipLine(ref v41, ref v42, model_mvp)) return;
                    drawline2d(
                        new Vector3D((1 + v41.X / v41.W) * render_width / 2, (1 - v41.Y / v41.W) * render_height / 2, v41.Z / v41.W),
                        new Vector3D((1 + v42.X / v42.W) * render_width / 2, (1 - v42.Y / v42.W) * render_height / 2, v42.Z / v42.W));
                };
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    DrawHelp.DrawParametricUV(drawline3d, uv.GetPointUV);
                    continue;
                }
                IParametricUVW uvw = transformedobject.Node.Primitive as IParametricUVW;
                if (uvw != null)
                {
                    DrawHelp.DrawParametricUVW(drawline3d, uvw.GetPointUVW);
                    continue;
                }
                Mesh mesh = transformedobject.Node.Primitive as Mesh;
                if (mesh != null)
                {
                    var v = mesh.Vertices;
                    foreach (var t in mesh.Triangles)
                    {
                        drawline3d(v[t.Index0], v[t.Index1]);
                        drawline3d(v[t.Index1], v[t.Index2]);
                        drawline3d(v[t.Index2], v[t.Index0]);
                    }
                    continue;
                }
                MeshBVH meshbvh = transformedobject.Node.Primitive as MeshBVH;
                if (meshbvh != null)
                {
                    var nodes_with_triangles = MeshBVH.EnumerateNodes(meshbvh)
                        .Where(x => x.Triangles != null);
                    foreach (var node in nodes_with_triangles)
                    {
                        var lines = new[]
                        {
                            new[] {0,0,0}, new[] {1,0,0},
                            new[] {0,0,0}, new[] {0,1,0},
                            new[] {0,0,0}, new[] {0,0,1},
                            new[] {1,0,0}, new[] {1,1,0},
                            new[] {1,0,0}, new[] {1,0,1},
                            new[] {0,1,0}, new[] {1,1,0},
                            new[] {0,1,0}, new[] {0,1,1},
                            new[] {1,1,0}, new[] {1,1,1},
                            new[] {0,0,1}, new[] {1,0,1},
                            new[] {0,0,1}, new[] {0,1,1},
                            new[] {1,0,1}, new[] {1,1,1},
                            new[] {0,1,1}, new[] {1,1,1},
                        };
                        for (int line = 0; line < lines.Length; line += 2)
                        {
                            var i0 = lines[line + 0];
                            var i1 = lines[line + 1];
                            var p0 = new Vector3D(i0[0] == 0 ? node.Bound.Min.X : node.Bound.Max.X, i0[1] == 0 ? node.Bound.Min.Y : node.Bound.Max.Y, i0[2] == 0 ? node.Bound.Min.Z : node.Bound.Max.Z);
                            var p1 = new Vector3D(i1[0] == 0 ? node.Bound.Min.X : node.Bound.Max.X, i1[1] == 0 ? node.Bound.Min.Y : node.Bound.Max.Y, i1[2] == 0 ? node.Bound.Min.Z : node.Bound.Max.Z);
                            drawline3d(p0, p1);
                        }
                        foreach (var t in node.Triangles)
                        {
                            drawline3d(t.P0, t.P1);
                            drawline3d(t.P1, t.P2);
                            drawline3d(t.P2, t.P0);
                        }
                    }
                    continue;
                }
            }
        }
        #endregion
        #region - Section : Phase 3 - Rasterized Rendering (Reference) -
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
                    foreach (var tri in ClipHelp.ClipTriangle4D(new Triangle4D(p1, p2, p3)))
                    {
                        Vector4D[] v3 = { tri.P0, tri.P1, tri.P2 };
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
        #endregion
    }
}