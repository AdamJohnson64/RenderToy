////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.PipelineModel
{
    static partial class PrimitiveAssembly
    {
        /// <summary>
        /// Convert an input scene into a triangle list.
        /// </summary>
        /// <param name="scene">The source scene.</param>
        /// <returns>A stream of triangles representing the scene.</returns>
        public static IEnumerable<Vector3D> CreateTriangles(IScene scene)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D modeltransform = transformedobject.Transform;
                foreach (var x in CreateTriangles(transformedobject.Node.Primitive))
                {
                    yield return MathHelp.TransformPoint(modeltransform, x);
                }
            }
        }
        /// <summary>
        /// Create triangles representing a primitive.
        /// </summary>
        /// <param name="prim">The primitive.</param>
        /// <returns>A stream of triangles describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateTriangles(IPrimitive prim)
        {
            IParametricUV uv = prim as IParametricUV;
            if (uv != null)
            {
                return CreateTriangles(uv);
            }
            Mesh mesh = prim as Mesh;
            if (mesh != null)
            {
                return CreateTriangles(mesh);
            }
            MeshBVH meshbvh = prim as MeshBVH;
            if (meshbvh != null)
            {
                return CreateTriangles(meshbvh);
            }
            return new Vector3D[] { };
        }
        /// <summary>
        /// Create triangles representing a UV parametric surface.
        /// </summary>
        /// <param name="uv">The UV parametric surface.</param>
        /// <returns>A stream of triangles describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateTriangles(IParametricUV uv)
        {
            return CreateTriangles(uv, 10, 10);
        }
        /// <summary>
        /// Create triangles representing a UV parametric surface.
        /// </summary>
        /// <param name="uv">The UV parametric surface.</param>
        /// <returns>A stream of triangles describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateTriangles(IParametricUV uv, int USEGMENTS, int VSEGMENTS)
        {
            for (int u = 0; u < USEGMENTS; ++u)
            {
                for (int v = 0; v < VSEGMENTS; ++v)
                {
                    Vector3D p300 = uv.GetPointUV((u + 0.0) / USEGMENTS, (v + 0.0) / VSEGMENTS);
                    Vector3D p310 = uv.GetPointUV((u + 1.0) / USEGMENTS, (v + 0.0) / VSEGMENTS);
                    Vector3D p301 = uv.GetPointUV((u + 0.0) / USEGMENTS, (v + 1.0) / VSEGMENTS);
                    Vector3D p311 = uv.GetPointUV((u + 1.0) / USEGMENTS, (v + 1.0) / VSEGMENTS);
                    yield return p300; yield return p310; yield return p311;
                    yield return p311; yield return p301; yield return p300;
                }
            }
        }
        /// <summary>
        /// Create triangles representing a UVW parametric volume.
        /// </summary>
        /// <param name="uvw">The UVW parametric volume.</param>
        /// <returns>A stream of triangles describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateTriangles(IParametricUVW uvw)
        {
            return new Vector3D[] { };
        }
        /// <summary>
        /// Create triangles representing a simple mesh.
        /// </summary>
        /// <param name="mesh">The mesh.</param>
        /// <returns>A stream of triangles describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateTriangles(Mesh mesh)
        {
            var v = mesh.Vertices.GetVertices();
            foreach (var t in mesh.Vertices.GetIndices())
            {
                yield return v[t];
            }
        }
        /// <summary>
        /// Create triangles representing a BVH split mesh.
        /// </summary>
        /// <param name="meshbvh">The mesh.</param>
        /// <returns>A stream of triangles describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateTriangles(MeshBVH meshbvh)
        {
            var nodes_with_triangles = MeshBVH.EnumerateNodes(meshbvh)
                .Where(x => x.Triangles != null);
            foreach (var node in nodes_with_triangles)
            {
                foreach (var t in node.Triangles)
                {
                    yield return t.P0;
                    yield return t.P1;
                    yield return t.P2;
                }
            }
        }
    }
}