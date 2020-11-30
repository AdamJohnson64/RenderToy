using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.Primitives;
using System.Collections.Generic;
using System.Linq;

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
            public Vector3D Tangent;
            public Vector3D Bitangent;
        }
        public static IEnumerable<VertexDX> CreateTrianglesDX(IPrimitive prim)
        {
            IParametricUV uv = prim as IParametricUV;
            if (uv != null)
            {
                return CreateTrianglesDX(uv);
            }
            Mesh mesh = prim as Mesh;
            if (mesh != null)
            {
                return CreateTrianglesDX(mesh);
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
                    Vector3D n300 = uv.GetNormalUV(uv00.X, uv00.Y);
                    Vector3D n310 = uv.GetNormalUV(uv10.X, uv10.Y);
                    Vector3D n301 = uv.GetNormalUV(uv01.X, uv01.Y);
                    Vector3D n311 = uv.GetNormalUV(uv11.X, uv11.Y);
                    Vector3D t300 = uv.GetTangentUV(uv00.X, uv00.Y);
                    Vector3D t310 = uv.GetTangentUV(uv10.X, uv10.Y);
                    Vector3D t301 = uv.GetTangentUV(uv01.X, uv01.Y);
                    Vector3D t311 = uv.GetTangentUV(uv11.X, uv11.Y);
                    Vector3D b300 = uv.GetBitangentUV(uv00.X, uv00.Y);
                    Vector3D b310 = uv.GetBitangentUV(uv10.X, uv10.Y);
                    Vector3D b301 = uv.GetBitangentUV(uv01.X, uv01.Y);
                    Vector3D b311 = uv.GetBitangentUV(uv11.X, uv11.Y);
                    VertexDX v00 = new VertexDX { Position = p300, Normal = n300, TexCoord = uv00, Diffuse = NormalColor(n300), Tangent = t300, Bitangent = b300 };
                    VertexDX v10 = new VertexDX { Position = p310, Normal = n310, TexCoord = uv10, Diffuse = NormalColor(n310), Tangent = t310, Bitangent = b310 };
                    VertexDX v01 = new VertexDX { Position = p301, Normal = n301, TexCoord = uv01, Diffuse = NormalColor(n301), Tangent = t301, Bitangent = b301 };
                    VertexDX v11 = new VertexDX { Position = p311, Normal = n311, TexCoord = uv11, Diffuse = NormalColor(n311), Tangent = t311, Bitangent = b311 };
                    yield return v00; yield return v10; yield return v11;
                    yield return v11; yield return v01; yield return v00;
                }
            }
        }
        static uint NormalColor(Vector3D normal)
        {
            // TODO: This is a useful test, we might want to resurrect it later.
            return 0xFFFFFFFF;
            /*
            var v4 = Transformation.Vector3ToVector4(normal);
            v4.X = (v4.X + 1) * 0.5;
            v4.Y = (v4.Y + 1) * 0.5;
            v4.Z = (v4.Z + 1) * 0.5;
            return Rasterization.ColorToUInt32(v4);
            */
        }
        /// <summary>
        /// Create triangles representing a simple mesh.
        /// </summary>
        /// <param name="mesh">The mesh.</param>
        /// <returns>A stream of triangles describing the surface of this primitive.</returns>
        public static IEnumerable<VertexDX> CreateTrianglesDX(Mesh mesh)
        {
            var vpos = mesh.Vertices?.GetVertices();
            var ipos = mesh.Vertices?.GetIndices();
            var vnor = mesh.Normals?.GetVertices();
            var inor = mesh.Normals?.GetIndices();
            var vtex = mesh.TexCoords?.GetVertices();
            var itex = mesh.TexCoords?.GetIndices();
            var vtan = mesh.Tangents?.GetVertices();
            var itan = mesh.Tangents?.GetIndices();
            var vbit = mesh.Bitangents?.GetVertices();
            var ibit = mesh.Bitangents?.GetIndices();
            int trianglecount = ipos.Count;
            for (int i = 0; i < trianglecount; ++i)
            {
                var v = new VertexDX { Diffuse = 0xFFFFFFFF };
                if (vpos != null && ipos != null) v.Position = vpos[ipos[i]];
                if (vnor != null && inor != null) v.Normal = vnor[inor[i]];
                if (vtex != null && itex != null) v.TexCoord = vtex[itex[i]];
                if (vtan != null && itan != null) v.Tangent = vtan[itan[i]];
                if (vbit != null && ibit != null) v.Bitangent = vbit[ibit[i]];
                yield return v;
            }
        }
    }
}
