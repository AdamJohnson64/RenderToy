////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using System.Collections.Generic;

namespace RenderToy.PipelineModel
{
    static partial class PrimitiveAssembly
    {
        /// <summary>
        /// Convert an input scene into a point list.
        /// </summary>
        /// <param name="scene">The source scene.</param>
        /// <returns>A stream of vertices.</returns>
        public static IEnumerable<Vector3D> CreatePoints(IEnumerable<TransformedObject> scene)
        {
            foreach (var transformedobject in scene)
            {
                Matrix3D modeltransform = transformedobject.Transform;
                foreach (var x in CreatePoints(transformedobject.NodePrimitive))
                {
                    yield return MathHelp.TransformPoint(modeltransform, x);
                }
            }
        }
        /// <summary>
        /// Create points representing a primitive.
        /// </summary>
        /// <param name="prim">The primitive.</param>
        /// <returns>A stream of points describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreatePoints(IPrimitive prim)
        {
            IParametricUV uv = prim as IParametricUV;
            if (uv != null)
            {
                return CreatePoints(uv);
            }
            IParametricUVW uvw = prim as IParametricUVW;
            if (uvw != null)
            {
                return CreatePoints(uvw);
            }
            Mesh mesh = prim as Mesh;
            if (mesh != null)
            {
                return CreatePoints(mesh);
            }
            return new Vector3D[] { };
        }
        /// <summary>
        /// Create points representing a UV parametric surface.
        /// </summary>
        /// <param name="uv">The UV parametric surface.</param>
        /// <returns>A stream of points describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreatePoints(IParametricUV uv)
        {
            int USEGMENTS = 20;
            int VSEGMENTS = 20;
            // Simply move some number of steps across u and v and draw the points in space.
            for (int u = 0; u <= USEGMENTS; ++u)
            {
                for (int v = 0; v <= VSEGMENTS; ++v)
                {
                    // Determine the point and draw it; easy.
                    yield return uv.GetPointUV((double)u / USEGMENTS, (double)v / VSEGMENTS);
                }
            }
        }
        /// <summary>
        /// Create points representing a UVW parametric surface.
        /// </summary>
        /// <param name="uvw">The UVW parametric surface.</param>
        /// <returns>A stream of points describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreatePoints(IParametricUVW uvw)
        {
            int USEGMENTS = 20;
            int VSEGMENTS = 20;
            int WSEGMENTS = 20;
            // Simply move some number of steps across u and v and draw the points in space.
            for (int u = 0; u <= USEGMENTS; ++u)
            {
                for (int v = 0; v <= VSEGMENTS; ++v)
                {
                    for (int w = 0; w <= VSEGMENTS; ++w)
                    {
                        // Determine the point and draw it; easy.
                        yield return uvw.GetPointUVW((double)u / USEGMENTS, (double)v / VSEGMENTS, (double)w / WSEGMENTS);
                    }
                }
            }
        }
        /// <summary>
        /// Create points representing a simple mesh.
        /// </summary>
        /// <param name="mesh">The mesh.</param>
        /// <returns>A stream of points describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreatePoints(Mesh mesh)
        {
            return mesh.Vertices.GetVertices();
        }
        /// <summary>
        /// Create points representing a BVH split mesh.
        /// </summary>
        /// <param name="meshbvh">The mesh.</param>
        /// <returns>A stream of points describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreatePoints(MeshBVH meshbvh)
        {
            return new Vector3D[] { };
        }
    }
}