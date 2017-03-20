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
    }
}