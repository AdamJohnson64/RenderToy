////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.PipelineModel;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace RenderToy.WPF
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
                var A = PrimitiveAssembly.CreateTriangles(transformedobject.Node.Primitive);
                var B = Transformation.Vector3ToVector4(A);
                var C = Transformation.Transform(B, transformedobject.Transform * mvp);
                var D = Clipping.ClipTriangle(C);
                var E = Transformation.HomogeneousDivide(D);
                var F = E.Select(i => new XYZWDiffuse { X = (float)i.X, Y = (float)i.Y, Z = (float)i.Z, W = (float)i.W, Color = Rasterization.ColorToUInt32(transformedobject.Node.WireColor) });
                var vertexbuffer = F.ToArray();
                d3dsurface.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.D3DPT_TRIANGLELIST, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(XYZWDiffuse)));
            }
            d3dsurface.EndScene();
            d3dsurface.CopyTo(bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        struct XYZWDiffuse
        {
            public float X, Y, Z, W;
            public uint Color;
        };
        #endregion
    }
}
