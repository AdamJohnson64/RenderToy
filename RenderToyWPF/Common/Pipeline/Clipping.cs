﻿using RenderToy.Math;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.PipelineModel
{
    public static partial class Clipping
    {
        #region - Section : World Clip (Triangles) -
        /// <summary>
        /// Clip a single 3D triangle against an arbitrary plane.
        /// </summary>
        /// <param name="triangle">The 3D triangle to be clipped.</param>
        /// <param name="plane_normal">The normal of the plane to clip with.</param>
        /// <param name="plane_distance">The offset of the plane along its normal.</param>
        /// <returns>A list of 3D triangle fragments for the resulting clipped 3D triangle.</returns>
        public static IEnumerable<Vector3D> ClipTriangle(IEnumerable<Vector3D> triangles, Vector3D plane_normal, double plane_distance)
        {
            var iter = triangles.GetEnumerator();
            var p = new Vector3D[3];
            while (iter.MoveNext())
            {
                // Gather up the triangle points.
                p[0] = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                p[1] = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                p[2] = iter.Current;
                // Process for clipping.
                var sides = p.Select(x => MathHelp.Dot(x, plane_normal) - plane_distance);
                // Get the side for all points (inside or outside).
                var outside = sides
                    .Select((x, i) => new { index = i, side = x })
                    .Where(x => x.side > 0)
                    .ToArray();
                // Get the side for all points (inside or outside).
                var inside = sides
                    .Select((x, i) => new { index = i, side = x })
                    .Where(x => x.side <= 0)
                    .ToArray();
                // All points are clipped; trivially reject the whole triangle.
                if (outside.Length == 0 && inside.Length == 3)
                {
                    continue;
                }
                // If one point is clipped then emit the single remaining triangle.
                if (outside.Length == 1 && inside.Length == 2)
                {
                    var p1 = p[outside[0].index];
                    var p2 = p[inside[0].index];
                    var p3 = p[inside[1].index];
                    var pi1 = CalculateIntersectionPointLine(p1, p2, plane_normal, plane_distance);
                    var pi2 = CalculateIntersectionPointLine(p3, p1, plane_normal, plane_distance);
                    yield return p1; yield return pi1; yield return pi2;
                    continue;
                }
                // If two points are clipped then emit two triangles.
                if (outside.Length == 2 && inside.Length == 1)
                {
                    int first = outside[0].index;
                    var p1 = p[outside[0].index];
                    var p2 = p[outside[1].index];
                    var p3 = p[inside[0].index];
                    var p2i = CalculateIntersectionPointLine(p2, p3, plane_normal, plane_distance);
                    var p3i = CalculateIntersectionPointLine(p3, p1, plane_normal, plane_distance);
                    yield return p1; yield return p2; yield return p2i;
                    yield return p2i; yield return p3i; yield return p1;
                    continue;
                }
                // All points are unclipped; trivially accept this triangle.
                if (outside.Length == 3 && inside.Length == 0)
                {
                    yield return p[0]; yield return p[1]; yield return p[2]; continue;
                }
            }
        }
        /// <summary>
        /// Calculate the lambda for the intersection of a 3D line segment with a plane.
        /// </summary>
        /// <param name="p1">The first point of the 3D line segment.</param>
        /// <param name="p2">The second point of the 3D line segment.</param>
        /// <param name="plane_normal">The normal of the plane to clip with.</param>
        /// <param name="plane_distance">The offset of the plane along its normal.</param>
        /// <returns>A distance value (lambda) between P1 and P2 at which the 3D line intersects the plane.</returns>
        static double CalculateIntersectionDistanceLine(Vector3D p1, Vector3D p2, Vector3D plane_normal, double plane_distance)
        {
            return CalculateIntersectionDistanceRay(p1, p2 - p1, plane_normal, plane_distance);
        }
        /// <summary>
        /// Calculate the intersection point for the intersection of a 3D line segment with a plane.
        /// </summary>
        /// <param name="p1">The first point of the line segment.</param>
        /// <param name="p2">The second point of the line segment.</param>
        /// <param name="plane_normal">The normal of the plane to clip with.</param>
        /// <param name="plane_distance">The offset of the plane along its normal.</param>
        /// <returns>The point at which the 3D line segment intersects the plane.</returns>
        static Vector3D CalculateIntersectionPointLine(Vector3D p1, Vector3D p2, Vector3D plane_normal, double plane_distance)
        {
            return p1 + MathHelp.Multiply(p2 - p1, CalculateIntersectionDistanceLine(p1, p2, plane_normal, plane_distance));
        }
        /// <summary>
        /// Calculate the lambda for the intersection of a 3D ray with a plane.
        /// </summary>
        /// <param name="origin">The origin of the 3D ray.</param>
        /// <param name="direction">The direction of the 3D ray.</param>
        /// <param name="plane_normal">The normal of the plane to clip with.</param>
        /// <param name="plane_distance">The offset of the plane along its normal.</param>
        /// <returns>A distance value (lambda) between P1 and P2 at which the 3D ray intersects the plane.</returns>
        static double CalculateIntersectionDistanceRay(Vector3D origin, Vector3D direction, Vector3D plane_normal, double plane_distance)
        {
            // Compute the intersection with the clip plane.
            return (plane_distance - MathHelp.Dot(plane_normal, origin)) / MathHelp.Dot(plane_normal, direction);
        }
        #endregion
        #region - Section : Homogeneous Clip (Common) -
        /// <summary>
        /// Calculate the lambda for the intersection of a 4D line segment with a plane.
        /// </summary>
        /// <param name="p1">The first point of the 4D line segment.</param>
        /// <param name="p2">The second point of the 4D line segment.</param>
        /// <param name="plane">The 4D plane to clip with.</param>
        /// <returns>A distance value (lambda) between P1 and P2 at which the 4D line intersects the plane.</returns>
        static double CalculateIntersectionDistanceLine(Vector4D p1, Vector4D p2, Vector4D plane)
        {
            return CalculateIntersectionDistanceRay(p1, p2 - p1, plane);
        }
        /// <summary>
        /// Calculate the intersection point for the intersection of a 4D line segment with a plane.
        /// </summary>
        /// <param name="p1">The first point of the 4D line segment.</param>
        /// <param name="p2">The second point of the 4D line segment.</param>
        /// <param name="plane">The 4D plane to clip with.</param>
        /// <returns>The point at which the 4D line segment intersects the plane.</returns>
        static Vector4D CalculateIntersectionPointLine(Vector4D p1, Vector4D p2, Vector4D plane)
        {
            return p1 + MathHelp.Multiply(p2 - p1, CalculateIntersectionDistanceLine(p1, p2, plane));
        }
        /// <summary>
        /// Calculate the lambda for the intersection of a 4D ray with a plane.
        /// </summary>
        /// <param name="origin">The origin of the 4D ray.</param>
        /// <param name="direction">The direction of the 4D ray.</param>
        /// <param name="plane">The 4D plane to clip with.</param>
        /// <returns>A distance value (lambda) between P1 and P2 at which the 4D ray intersects the plane.</returns>
        static double CalculateIntersectionDistanceRay(Vector4D origin, Vector4D direction, Vector4D plane)
        {
            // Compute the intersection with the clip plane.
            return -MathHelp.Dot(plane, origin) / MathHelp.Dot(plane, direction);
        }
        #endregion
        #region - Section : Homogeneous Clip (Points) -
        /// <summary>
        /// Clip a list of vertices.
        /// 
        /// Internally this simply omits points for which w≤0.
        /// </summary>
        /// <param name="points">The vertex source to be clipped.</param>
        /// <returns>A filtered list of vertices for which no vertex has w≤0.</returns>
        public static IEnumerable<Vector4D> ClipPoint(IEnumerable<Vector4D> points)
        {
            return points.Where(v => v.W > 0);
        }
        #endregion
        #region - Section : Homogeneous Clip (Lines) -
        /// <summary>
        /// Clip a list of 4D lines.
        /// </summary>
        /// <param name="lines">The line source to be clipped.</param>
        /// <returns>A stream of line segments completely clipped by and contained in clip space.</returns>
        public static IEnumerable<Vector4D> ClipLine(IEnumerable<Vector4D> lines)
        {
            var iter = lines.GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext())
                {
                    yield break;
                }
                var P1 = iter.Current;
                if (Clipping.ClipLine(ref P0, ref P1))
                {
                    yield return P0;
                    yield return P1;
                }
            }
        }
        /// <summary>
        /// Homogeneous clip a clip-space line segment.
        /// </summary>
        /// <param name="p1">The clip-space starting position.</param>
        /// <param name="p2">The clip-space ending position.</param>
        /// <returns>True if any part of the line remains, false if it was completely clipped away.</returns>
        public static bool ClipLine(ref Vector4D p1, ref Vector4D p2)
        {
            // Clip to clip-space near (z=0).
            if (!ClipLine(ref p1, ref p2, new Vector4D(0, 0, 1, 0))) return false;
            // Clip to clip-space far (z=w, z/w=1).
            if (!ClipLine(ref p1, ref p2, new Vector4D(0, 0, -1, 1))) return false;
            // Clip to clip-space right (-x+w=0, x/w=1).
            if (!ClipLine(ref p1, ref p2, new Vector4D(-1, 0, 0, 1))) return false;
            // Clip to clip-space left (x+w=0, -x/w=1).
            if (!ClipLine(ref p1, ref p2, new Vector4D(1, 0, 0, 1))) return false;
            // Clip to clip-space top (-y+w=0, y/w=1).
            if (!ClipLine(ref p1, ref p2, new Vector4D(0, -1, 0, 1))) return false;
            // Clip to clip-space bottom (y+w=0, -y/w=1).
            if (!ClipLine(ref p1, ref p2, new Vector4D(0, 1, 0, 1))) return false;
            return true;
        }
        /// <summary>
        /// Homogeneous clip a clip-space line segment against a single clip-space plane.
        /// </summary>
        /// <param name="p1">The clip-space starting position.</param>
        /// <param name="p2">The clip-space ending position.</param>
        /// <param name="plane">The Ax+By+Cz+Dw=0 definition of the clip plane.</param>
        /// <returns>True if any part of the line remains, false if it was completely clipped away.</returns>
        static bool ClipLine(ref Vector4D p1, ref Vector4D p2, Vector4D plane)
        {
            // Determine which side of the plane these points reside.
            double side_p1 = MathHelp.Dot(p1, plane);
            double side_p2 = MathHelp.Dot(p2, plane);
            // If the line is completely behind the clip plane then reject it immediately.
            if (side_p1 <= 0 && side_p2 <= 0) return false;
            // If the line is completely in front of the clip plane then accept it immediately.
            if (side_p1 >= 0 && side_p2 >= 0) return true;
            // Otherwise the line straddles the clip plane; clip as appropriate.
            // Construct a line segment to clip.
            Vector4D line_org = p1;
            Vector4D line_dir = p2 - p1;
            // Compute the intersection with the clip plane.
            double lambda = CalculateIntersectionDistanceRay(line_org, line_dir, plane);
            // If the intersection lies in the line segment then clip.
            if (lambda > 0 && lambda < 1)
            {
                // If P1 was behind the plane them move it to the intersection point.
                if (side_p1 <= 0) p1 = line_org + MathHelp.Multiply(line_dir, lambda);
                // If P2 was behind the plane then move it to the intersection point.
                if (side_p2 <= 0) p2 = line_org + MathHelp.Multiply(line_dir, lambda);
            }
            return true;
        }
        #endregion
        #region - Section : Homogeneous Clip (Triangles) -
        /// <summary>
        /// Clip a list of 4D triangles against the clip space box (all planes).
        /// </summary>
        /// <param name="triangles">The 4D triangle source to be clipped.</param>
        /// <returns>A stream of 4D triangles completely clipped by and contained in clip space.</returns>
        public static IEnumerable<Vector4D> ClipTriangle(IEnumerable<Vector4D> triangles)
        {
            var result = ClipTriangle(triangles, new Vector4D(0, 0, 1, 0));
            result = ClipTriangle(result, new Vector4D(0, 0, -1, 1));
            result = ClipTriangle(result, new Vector4D(-1, 0, 0, 1));
            result = ClipTriangle(result, new Vector4D(1, 0, 0, 1));
            result = ClipTriangle(result, new Vector4D(0, -1, 0, 1));
            result = ClipTriangle(result, new Vector4D(0, 1, 0, 1));
            return result.ToArray();
        }
        /// <summary>
        /// Clip a list of 4D triangles against a plane.
        /// </summary>
        /// <param name="tri">The triangle source to be clipped.</param>
        /// <param name="plane">The plane to clip the triangles against.</param>
        /// <returns>A stream of triangles clipped by the plane.</returns>
        public static IEnumerable<Vector4D> ClipTriangle(IEnumerable<Vector4D> tri, Vector4D plane)
        {
            var iter = tri.GetEnumerator();
            var p = new Vector4D[3];
            while (iter.MoveNext())
            {
                p[0] = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                p[1] = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                p[2] = iter.Current;
                var sides = p.Select(x => MathHelp.Dot(x, plane));
                // Get the side for all points (inside or outside).
                var outside = sides
                    .Select((x, i) => new { index = i, side = x })
                    .Where(x => x.side > 0)
                    .ToArray();
                // Get the side for all points (inside or outside).
                var inside = sides
                    .Select((x, i) => new { index = i, side = x })
                    .Where(x => x.side <= 0)
                    .ToArray();
                // All points are clipped; trivially reject the whole triangle.
                if (outside.Length == 0 && inside.Length == 3)
                {
                    continue;
                }
                // If one point is clipped then emit the single remaining triangle.
                if (outside.Length == 1 && inside.Length == 2)
                {
                    var p1 = p[outside[0].index];
                    var p2 = p[inside[0].index];
                    var p3 = p[inside[1].index];
                    var pi1 = CalculateIntersectionPointLine(p1, p2, plane);
                    var pi2 = CalculateIntersectionPointLine(p3, p1, plane);
                    yield return p1; yield return pi1; yield return pi2;
                    continue;
                }
                // If two points are clipped then emit two triangles.
                if (outside.Length == 2 && inside.Length == 1)
                {
                    int first = outside[0].index;
                    var p1 = p[outside[0].index];
                    var p2 = p[outside[1].index];
                    var p3 = p[inside[0].index];
                    var p2i = CalculateIntersectionPointLine(p2, p3, plane);
                    var p3i = CalculateIntersectionPointLine(p3, p1, plane);
                    yield return p1; yield return p2; yield return p2i;
                    yield return p2i; yield return p3i; yield return p1;
                    continue;
                }
                // All points are unclipped; trivially accept this triangle.
                if (outside.Length == 3 && inside.Length == 0)
                {
                    yield return p[0]; yield return p[1]; yield return p[2];
                    continue;
                }
            }
        }
        #endregion
    }
}