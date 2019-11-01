////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Expressions;
using RenderToy.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;

namespace RenderToy.PipelineModel
{
    /// <summary>
    /// Pipeline functions for assembling render pipelines.
    /// 
    /// These functions can be chained arbitrarily to assemble complete or
    /// partial pipelines. Components can be omitted to test cases such as
    /// (e.g.) a missing clipper unit or different transform behaviors.
    /// </summary>
    public static partial class Transformation
    {
        /// <summary>
        /// Perform a homogeneous divide on a 4D vector (for readability only).
        /// </summary>
        /// <param name="vertex">The input vector.</param>
        /// <returns>A vector of the form [x/w,y/y,z/w,1].</returns>
        static Expression<Func<Vector4D, Vector4D>> HomogeneousDivideFn = (vertex) => MathHelp.Multiply(1.0 / vertex.W, vertex);
        public static ExpressionFlatten<Func<Vector4D, Vector4D>> HomogeneousDivide = HomogeneousDivideFn.ReplaceCalls().Rename("HomogeneousDivide").Flatten();
        /// <summary>
        /// Perform a homogeneous divide on a vertex stream.
        /// 
        /// [x,y,z,w] will be transformed to [x/w,y/w,z/w,1] for all vertices.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed</param>
        /// <returns>A stream of post-homogeneous-divide vertices.</returns>
        public static IEnumerable<Vector4D> HomogeneousDivideAll(IEnumerable<Vector4D> vertices) => vertices.Select(v => HomogeneousDivide.Call(v));
        /// <summary>
        /// Transform a stream of vertices by an arbitrary 4D matrix.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <param name="transform">The transform to apply to each vertex.</param>
        /// <returns>A stream of transformed vertices.</returns>
        public static IEnumerable<Vector4D> TransformAll(IEnumerable<Vector4D> vertices, Matrix3D transform) => vertices.Select(v => MathHelp.Transform(transform, v));
        /// <summary>
        /// Transform a homogeneous vertex into screen space with the supplied dimensions.
        /// </summary>
        /// <param name="vertex">The vertex to be transformed.</param>
        /// <param name="width">The width of the screen area in pixels.</param>
        /// <param name="height">The height of the screen area in pixels.</param>
        /// <returns>The vertex transformed into screen space.</returns>
        static Expression<Func<Vector4D, double, double, Vector4D>> TransformToScreenFn = (vertex, width, height) => new Vector4D((vertex.X + 1) * width / 2, (1 - vertex.Y) * height / 2, vertex.Z, vertex.W);
        public static ExpressionFlatten<Func<Vector4D, double, double, Vector4D>> TransformToScreen = TransformToScreenFn.Rename("TransformToScreen").Flatten();
        /// <summary>
        /// Transform a list of vertices into screen space.
        /// </summary>
        /// <param name="vertices">The vertices to be transformed.</param>
        /// <param name="width">The width of the screen area in pixels.</param>
        /// <param name="height">The height of the screen area in pixels.</param>
        /// <returns>A stream of screen-space transformed vertices.</returns>
        public static IEnumerable<Vector4D> TransformToScreenAll(IEnumerable<Vector4D> vertices, double width, double height) => vertices.Select(v => TransformToScreen.Call(v, width, height));
        /// <summary>
        /// Up-cast a stream of 3D vectors to a stream of 4D vectors with w=1.
        /// </summary>
        /// <param name="vertices">A stream of 3D vectors.</param>
        /// <returns>A stream of 4D vectors with w=1.</returns>
        static Expression<Func<Vector3D, Vector4D>> Vector3ToVector4Fn = (vertex) => new Vector4D { X = vertex.X, Y = vertex.Y, Z = vertex.Z, W = 1 };
        public static ExpressionFlatten<Func<Vector3D, Vector4D>> Vector3ToVector4 = Vector3ToVector4Fn.Rename("Vector3ToVector4").Flatten();
        /// <summary>
        /// Cast a sequence of Vector3 points to their homogeneous representation [x,y,z,1].
        /// </summary>
        /// <param name="vertices">The vertices to cast.</param>
        /// <returns>A stream of homogeneous vertices expanded as [x,y,z,1].</returns>
        public static IEnumerable<Vector4D> Vector3ToVector4All(IEnumerable<Vector3D> vertices) => vertices.Select(v => Vector3ToVector4.Call(v));
    }
}