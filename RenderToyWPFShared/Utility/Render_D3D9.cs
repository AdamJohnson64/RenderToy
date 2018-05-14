////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.PipelineModel;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;

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
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv == null) continue;
                d3dsurface.SetColor(Rasterization.ColorToUInt32(transformedobject.Node.WireColor));
                var A = PipelineModel.PrimitiveAssembly.CreateTriangles(uv);
                var B = Transformation.Vector3ToVector4(A);
                var C = Transformation.Transform(B, transformedobject.Transform * mvp);
                var D = Clipping.ClipTriangle(C);
                var E = Transformation.HomogeneousDivide(D);
                var iter = E.GetEnumerator();
                while (iter.MoveNext())
                {
                    var P0 = iter.Current;
                    if (!iter.MoveNext()) break;
                    var P1 = iter.Current;
                    if (!iter.MoveNext()) break;
                    var P2 = iter.Current;
                    d3dsurface.DrawTriangle(
                        (float)P0.X, (float)P0.Y, (float)P0.Z, (float)P0.W,
                        (float)P1.X, (float)P1.Y, (float)P1.Z, (float)P1.W,
                        (float)P2.X, (float)P2.Y, (float)P2.Z, (float)P2.W);
                }
            }
            d3dsurface.EndScene();
            d3dsurface.CopyTo(bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        #endregion
    }
}
