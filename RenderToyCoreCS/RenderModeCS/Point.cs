////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using RenderToy.SceneGraph.Meshes;
using RenderToy.SceneGraph.Primitives;
using System;

namespace RenderToy
{
    public static partial class RenderModeCS
    {
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
    }
}