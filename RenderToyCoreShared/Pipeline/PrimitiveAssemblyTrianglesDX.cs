////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Primitives;
using RenderToy.Utility;
using System.Collections.Generic;

namespace RenderToy.PipelineModel
{
    static partial class PrimitiveAssembly
    {
        public struct VertexDX
        {
            public Vector3D Position;
            public Vector3D Normal;
            public uint Diffuse;
            public Vector2D TexCoord;
        }
        public static IEnumerable<VertexDX> CreateTrianglesDX(IPrimitive prim)
        {
            IParametricUV uv = prim as IParametricUV;
            if (uv != null)
            {
                return CreateTrianglesDX(uv);
            }
            return new VertexDX[] { };
        }
        public static IEnumerable<VertexDX> CreateTrianglesDX(IParametricUV uv)
        {
            int USEGMENTS = 10;
            int VSEGMENTS = 10;
            for (int u = 0; u < USEGMENTS; ++u)
            {
                for (int v = 0; v < VSEGMENTS; ++v)
                {
                    Vector2D uv00 = new Vector2D((u + 0.0) / USEGMENTS, (v + 0.0) / VSEGMENTS);
                    Vector2D uv10 = new Vector2D((u + 1.0) / USEGMENTS, (v + 0.0) / VSEGMENTS);
                    Vector2D uv01 = new Vector2D((u + 0.0) / USEGMENTS, (v + 1.0) / VSEGMENTS);
                    Vector2D uv11 = new Vector2D((u + 1.0) / USEGMENTS, (v + 1.0) / VSEGMENTS);
                    Vector3D p300 = uv.GetPointUV(uv00.X, uv00.Y);
                    Vector3D p310 = uv.GetPointUV(uv10.X, uv10.Y);
                    Vector3D p301 = uv.GetPointUV(uv01.X, uv01.Y);
                    Vector3D p311 = uv.GetPointUV(uv11.X, uv11.Y);
                    VertexDX v00 = new VertexDX { Position = p300, TexCoord = uv00 };
                    VertexDX v10 = new VertexDX { Position = p310, TexCoord = uv10 };
                    VertexDX v01 = new VertexDX { Position = p301, TexCoord = uv01 };
                    VertexDX v11 = new VertexDX { Position = p311, TexCoord = uv11 };
                    yield return v00; yield return v10; yield return v11;
                    yield return v11; yield return v01; yield return v00;
                }
            }
        }
    }
}
