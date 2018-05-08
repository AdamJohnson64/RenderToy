using System.Collections.Generic;
using System.Linq;

namespace RenderToy.PipelineModel
{
    /// <summary>
    /// Pipeline functions for assembling render pipelines.
    /// 
    /// These functions can be chained arbitrarily to assemble complete or
    /// partial pipelines. Components can be omitted to test cases such as
    /// (e.g.) a missing clipper unit or different transform behaviors.
    /// </summary>
    static partial class Transformation
    {
        /// <summary>
        /// Perform a homogeneous divide on a 4D vector (for readability only).
        /// </summary>
        /// <param name="vertex">The input vector.</param>
        /// <returns>A vector of the form [x/w,y/y,z/w,1].</returns>
        public static Vector4D HomogeneousDivide(Vector4D vertex)
        {
            return MathHelp.Multiply(1.0 / vertex.W, vertex);
        }
        /// <summary>
        /// Perform a homogeneous divide on a vertex stream.
        /// 
        /// [x,y,z,w] will be transformed to [x/w,y/w,z/w,1] for all vertices.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed</param>
        /// <returns>A stream of post-homogeneous-divide vertices.</returns>
        public static IEnumerable<Vector4D> HomogeneousDivide(IEnumerable<Vector4D> vertices)
        {
            return vertices.Select(v => HomogeneousDivide(v));
        }
        /// <summary>
        /// Transform a stream of vertices by an arbitrary 4D matrix.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <param name="transform">The transform to apply to each vertex.</param>
        /// <returns>A stream of transformed vertices.</returns>
        public static IEnumerable<Vector4D> Transform(IEnumerable<Vector4D> vertices, Matrix3D transform)
        {
            return vertices.Select(v => MathHelp.Transform(transform, v));
        }
        /// <summary>
        /// Transform a homogeneous vertex into screen space with the supplied dimensions.
        /// </summary>
        /// <param name="vertex">The vertex to be transformed.</param>
        /// <param name="width">The width of the screen area in pixels.</param>
        /// <param name="height">The height of the screen area in pixels.</param>
        /// <returns>The vertex transformed into screen space.</returns>
        public static Vector4D TransformToScreen(Vector4D vertex, double width, double height)
        {
            return new Vector4D((vertex.X + 1) * width / 2, (1 - vertex.Y) * height / 2, vertex.Z, vertex.W);
        }
        /// <summary>
        /// Transform a list of vertices into screen space.
        /// </summary>
        /// <param name="vertices">The vertices to be transformed.</param>
        /// <param name="width">The width of the screen area in pixels.</param>
        /// <param name="height">The height of the screen area in pixels.</param>
        /// <returns>A stream of screen-space transformed vertices.</returns>
        public static IEnumerable<Vector4D> TransformToScreen(IEnumerable<Vector4D> vertices, double width, double height)
        {
            return vertices.Select(v => TransformToScreen(v, width, height));
        }
        /// <summary>
        /// Up-cast a stream of 3D vectors to a stream of 4D vectors with w=1.
        /// </summary>
        /// <param name="vertices">A stream of 3D vectors.</param>
        /// <returns>A stream of 4D vectors with w=1.</returns>
        public static Vector4D Vector3ToVector4(Vector3D vertex)
        {
            return new Vector4D { X = vertex.X, Y = vertex.Y, Z = vertex.Z, W = 1 };
        }
        /// <summary>
        /// Cast a sequence of Vector3 points to their homogeneous representation [x,y,z,1].
        /// </summary>
        /// <param name="vertices">The vertices to cast.</param>
        /// <returns>A stream of homogeneous vertices expanded as [x,y,z,1].</returns>
        public static IEnumerable<Vector4D> Vector3ToVector4(IEnumerable<Vector3D> vertices)
        {
            return vertices.Select(v => Vector3ToVector4(v));
        }
    }
}