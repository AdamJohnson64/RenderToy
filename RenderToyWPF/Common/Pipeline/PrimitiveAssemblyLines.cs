using RenderToy.Linq;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.PipelineModel
{
    public static partial class PrimitiveAssembly
    {
        /// <summary>
        /// Convert an input scene into a wireframe line list.
        /// </summary>
        /// <param name="scene">The source scene.</param>
        /// <returns>A stream of colored line segments.</returns>
        public static IEnumerable<Vector3D> CreateLines(IEnumerable<TransformedObject> scene)
        {
            foreach (var transformedobject in scene)
            {
                Matrix3D modeltransform = transformedobject.Transform;
                uint color = Rasterization.ColorToUInt32(transformedobject.NodeWireColor);
                foreach (var x in CreateLines(transformedobject.NodePrimitive))
                {
                    yield return MathHelp.TransformPoint(modeltransform, x);
                }
            }
        }
        /// <summary>
        /// Create line segments representing a primitive.
        /// </summary>
        /// <param name="prim">The primitive.</param>
        /// <returns>A stream of line segments describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateLines(IPrimitive prim)
        {
            IParametricUV uv = prim as IParametricUV;
            if (uv != null)
            {
                return CreateLines(uv);
            }
            IParametricUVW uvw = prim as IParametricUVW;
            if (uvw != null)
            {
                return CreateLines(uvw);
            }
            Mesh mesh = prim as Mesh;
            if (mesh != null)
            {
                return CreateLines(mesh);
            }
            return new Vector3D[] { };
        }
        /// <summary>
        /// Create line segments for a UV parametric surface.
        /// </summary>
        /// <param name="uv">The UV parametric primitive.</param>
        /// <returns>A stream of line segments describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateLines(IParametricUV uv)
        {
            int USEGMENTS = 10;
            int VSEGMENTS = 10;
            for (int u = 0; u <= USEGMENTS; ++u)
            {
                for (int v = 0; v < VSEGMENTS; ++v)
                {
                    // Draw U Lines.
                    {
                        yield return uv.GetPointUV((u + 0.0) / USEGMENTS, (v + 0.0) / VSEGMENTS);
                        yield return uv.GetPointUV((u + 0.0) / USEGMENTS, (v + 1.0) / VSEGMENTS);
                    }
                    // Draw V Lines.
                    {
                        yield return uv.GetPointUV((v + 0.0) / VSEGMENTS, (u + 0.0) / USEGMENTS);
                        yield return uv.GetPointUV((v + 1.0) / VSEGMENTS, (u + 0.0) / USEGMENTS);
                    }
                }
            }
        }
        /// <summary>
        /// Create line segments for a UVW parametric surface.
        /// </summary>
        /// <param name="uvw">The UVW parametric primitive.</param>
        /// <returns>A stream of line segments describing the volume of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateLines(IParametricUVW uvw)
        {
            yield break;
            /*
            int USEGMENTS = 10;
            int VSEGMENTS = 10;
            int WSEGMENTS = 10;
            // Simply move some number of steps across u and v and draw the points in space.
            for (int u = 0; u <= USEGMENTS; ++u)
            {
                for (int v = 0; v <= VSEGMENTS; ++v)
                {
                    for (int w = 0; w <= WSEGMENTS; ++w)
                    {
                        // Determine the point and draw it; easy.
                        //Vector3D p = uvw.GetPointUVW((double)u / USEGMENTS, (double)v / VSEGMENTS, (double)w / WSEGMENTS);
                        //yield return new Vertex<Vector3D> { Position = p, Color = color };
                    }
                }
            }
            */
        }
        /// <summary>
        /// Create line segments for a simple mesh.
        /// </summary>
        /// <param name="mesh">The mesh primitive.</param>
        /// <returns>A stream of line segments describing the surface of this primitive.</returns>
        public static IEnumerable<Vector3D> CreateLines(Mesh mesh)
        {
            var v = mesh.Vertices.GetVertices();
            foreach (var t in LinqExtensions.Split3(mesh.Vertices.GetIndices()))
            {
                yield return v[t.Item1]; yield return v[t.Item2];
                yield return v[t.Item2]; yield return v[t.Item3];
                yield return v[t.Item3]; yield return v[t.Item1];
            }
        }
    }
}