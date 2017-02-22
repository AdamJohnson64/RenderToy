////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Linq;

namespace RenderToy
{
    public static class ClipHelp
    {
        #region - Section : World Clip (Triangles) -
        public static IEnumerable<Triangle3D> ClipTriangle3D(Triangle3D triangle, Vector3D plane_normal, double plane_distance)
        {
            var p = new[] { triangle.P0, triangle.P1, triangle.P2 };
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
                yield break;
            }
            // If one point is clipped then emit the single remaining triangle.
            if (outside.Length == 1 && inside.Length == 2)
            {
                var p1 = p[outside[0].index];
                var p2 = p[inside[0].index];
                var p3 = p[inside[1].index];
                var pi1 = CalculateIntersectionPointLine(p1, p2, plane_normal, plane_distance);
                var pi2 = CalculateIntersectionPointLine(p3, p1, plane_normal, plane_distance);
                yield return new Triangle3D(p1, pi1, pi2);
                yield break;
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
                yield return new Triangle3D(p1, p2, p2i);
                yield return new Triangle3D(p2i, p3i, p1);
                yield break;
            }
            // All points are unclipped; trivially accept this triangle.
            if (outside.Length == 3 && inside.Length == 0)
            {
                yield return triangle;
                yield break;
            }
        }
        static double CalculateIntersectionDistanceLine(Vector3D p1, Vector3D p2, Vector3D plane_normal, double plane_distance)
        {
            return CalculateIntersectionDistanceRay(p1, p2 - p1, plane_normal, plane_distance);
        }
        static Vector3D CalculateIntersectionPointLine(Vector3D p1, Vector3D p2, Vector3D plane_normal, double plane_distance)
        {
            return p1 + MathHelp.Multiply(p2 - p1, CalculateIntersectionDistanceLine(p1, p2, plane_normal, plane_distance));
        }
        static double CalculateIntersectionDistanceRay(Vector3D origin, Vector3D direction, Vector3D plane_normal, double plane_distance)
        {
            // Compute the intersection with the clip plane.
            return (plane_distance - MathHelp.Dot(plane_normal, origin)) / MathHelp.Dot(plane_normal, direction);
        }
        #endregion
        #region - Section : Homogeneous Clip (Common) -
        static double CalculateIntersectionDistanceLine(Vector4D p1, Vector4D p2, Vector4D plane)
        {
            return CalculateIntersectionDistanceRay(p1, p2 - p1, plane);
        }
        static Vector4D CalculateIntersectionPointLine(Vector4D p1, Vector4D p2, Vector4D plane)
        {
            return p1 + MathHelp.Multiply(p2 - p1, CalculateIntersectionDistanceLine(p1, p2, plane));
        }
        static double CalculateIntersectionDistanceRay(Vector4D origin, Vector4D direction, Vector4D plane)
        {
            // Compute the intersection with the clip plane.
            return -MathHelp.Dot(plane, origin) / MathHelp.Dot(plane, direction);
        }
        #endregion
        #region - Section : Homogeneous Clip (Lines) -
        public static bool TransformAndClipLine(ref Vector4D p1, ref Vector4D p2, Matrix3D mvp)
        {
            // Transform the supplied points into projection space.
            p1 = mvp.Transform(p1);
            p2 = mvp.Transform(p2);
            // Perform homogeneous space clipping.
            return ClipLine3D(ref p1, ref p2);
        }
        /// <summary>
        /// Homogeneous clip a clip-space line segment.
        /// </summary>
        /// <param name="p1">The clip-space starting position.</param>
        /// <param name="p2">The clip-space ending position.</param>
        /// <returns>True if any part of the line remains, false if it was completely clipped away.</returns>
        static bool ClipLine3D(ref Vector4D p1, ref Vector4D p2)
        {
            // Clip to clip-space near (z=0).
            if (!ClipLine3D(ref p1, ref p2, new Vector4D(0, 0, 1, 0))) return false;
            // Clip to clip-space far (z=w, z/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Vector4D(0, 0, -1, 1))) return false;
            // Clip to clip-space right (-x+w=0, x/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Vector4D(-1, 0, 0, 1))) return false;
            // Clip to clip-space left (x+w=0, -x/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Vector4D(1, 0, 0, 1))) return false;
            // Clip to clip-space top (-y+w=0, y/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Vector4D(0, -1, 0, 1))) return false;
            // Clip to clip-space bottom (y+w=0, -y/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Vector4D(0, 1, 0, 1))) return false;
            return true;
        }
        /// <summary>
        /// Homogeneous clip a clip-space line segment against a single clip-space plane.
        /// </summary>
        /// <param name="p1">The clip-space starting position.</param>
        /// <param name="p2">The clip-space ending position.</param>
        /// <param name="plane">The Ax+By+Cz+Dw=0 definition of the clip plane.</param>
        /// <returns>True if any part of the line remains, false if it was completely clipped away.</returns>
        static bool ClipLine3D(ref Vector4D p1, ref Vector4D p2, Vector4D plane)
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
        public static IEnumerable<Triangle4D> ClipTriangle4D(Triangle4D tri)
        {
            return ClipTriangle4D(new Triangle4D[] { tri });
        }
        static IEnumerable<Triangle4D> ClipTriangle4D(IEnumerable<Triangle4D> triangles)
        {
            return triangles
                .SelectMany(x => ClipTriangle4D(x, new Vector4D(0, 0, 1, 0)))
                .SelectMany(x => ClipTriangle4D(x, new Vector4D(0, 0, -1, 1)))
                .SelectMany(x => ClipTriangle4D(x, new Vector4D(-1, 0, 0, 1)))
                .SelectMany(x => ClipTriangle4D(x, new Vector4D(1, 0, 0, 1)))
                .SelectMany(x => ClipTriangle4D(x, new Vector4D(0, -1, 0, 1)))
                .SelectMany(x => ClipTriangle4D(x, new Vector4D(0, 1, 0, 1)))
                .ToArray();
        }
        static IEnumerable<Triangle4D> ClipTriangle4D(Triangle4D triangle, Vector4D plane)
        {
            var p = new[] { triangle.P0, triangle.P1, triangle.P2 };
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
                yield break;
            }
            // If one point is clipped then emit the single remaining triangle.
            if (outside.Length == 1 && inside.Length == 2)
            {
                var p1 = p[outside[0].index];
                var p2 = p[inside[0].index];
                var p3 = p[inside[1].index];
                var pi1 = CalculateIntersectionPointLine(p1, p2, plane);
                var pi2 = CalculateIntersectionPointLine(p3, p1, plane);
                yield return new Triangle4D(p1, pi1, pi2);
                yield break;
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
                yield return new Triangle4D(p1, p2, p2i);
                yield return new Triangle4D(p2i, p3i, p1);
                yield break;
            }
            // All points are unclipped; trivially accept this triangle.
            if (outside.Length == 3 && inside.Length == 0)
            {
                yield return triangle;
                yield break;
            }
        }
        #endregion
    }
}