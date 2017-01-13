////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Linq;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static class ClipHelp
    {
        #region - Section : Homogeneous Clip (Common) -
        static double CalculateIntersectionDistanceLine(Point4D p1, Point4D p2, Point4D plane)
        {
            return CalculateIntersectionDistanceRay(p1, p2 - p1, plane);
        }
        static Point4D CalculateIntersectionPointLine(Point4D p1, Point4D p2, Point4D plane)
        {
            return p1 + MathHelp.Multiply(p2 - p1, CalculateIntersectionDistanceLine(p1, p2, plane));
        }
        static double CalculateIntersectionDistanceRay(Point4D origin, Point4D direction, Point4D plane)
        {
            // Compute the intersection with the clip plane.
            return -MathHelp.Dot(plane, origin) / MathHelp.Dot(plane, direction);
        }
        #endregion
        #region - Section : Homogeneous Clip (Lines) -
        public static bool TransformAndClipLine(ref Point4D p1, ref Point4D p2, Matrix3D mvp)
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
        static bool ClipLine3D(ref Point4D p1, ref Point4D p2)
        {
            // Clip to clip-space near (z=0).
            if (!ClipLine3D(ref p1, ref p2, new Point4D(0, 0, 1, 0))) return false;
            // Clip to clip-space far (z=w, z/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Point4D(0, 0, -1, 1))) return false;
            // Clip to clip-space right (-x+w=0, x/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Point4D(-1, 0, 0, 1))) return false;
            // Clip to clip-space left (x+w=0, -x/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Point4D(1, 0, 0, 1))) return false;
            // Clip to clip-space top (-y+w=0, y/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Point4D(0, -1, 0, 1))) return false;
            // Clip to clip-space bottom (y+w=0, -y/w=1).
            if (!ClipLine3D(ref p1, ref p2, new Point4D(0, 1, 0, 1))) return false;
            return true;
        }
        /// <summary>
        /// Homogeneous clip a clip-space line segment against a single clip-space plane.
        /// </summary>
        /// <param name="p1">The clip-space starting position.</param>
        /// <param name="p2">The clip-space ending position.</param>
        /// <param name="plane">The Ax+By+Cz+Dw=0 definition of the clip plane.</param>
        /// <returns>True if any part of the line remains, false if it was completely clipped away.</returns>
        static bool ClipLine3D(ref Point4D p1, ref Point4D p2, Point4D plane)
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
            Point4D line_org = p1;
            Point4D line_dir = p2 - p1;
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
        public struct Triangle
        {
            public Point4D p1, p2, p3;
        }
        public static IEnumerable<Triangle> ClipTriangle3D(Triangle tri)
        {
            return ClipTriangle3D(new Triangle[] { tri });
        }
        static IEnumerable<Triangle> ClipTriangle3D(IEnumerable<Triangle> triangles)
        {
            return triangles
                .SelectMany(x => ClipTriangle3D(x, new Point4D(0, 0, 1, 0)))
                .SelectMany(x => ClipTriangle3D(x, new Point4D(0, 0, -1, 1)))
                .SelectMany(x => ClipTriangle3D(x, new Point4D(-1, 0, 0, 1)))
                .SelectMany(x => ClipTriangle3D(x, new Point4D(1, 0, 0, 1)))
                .SelectMany(x => ClipTriangle3D(x, new Point4D(0, -1, 0, 1)))
                .SelectMany(x => ClipTriangle3D(x, new Point4D(0, 1, 0, 1)))
                .ToArray();
        }
        static IEnumerable<Triangle> ClipTriangle3D(Triangle triangle, Point4D plane)
        {
            var p = new[] { triangle.p1, triangle.p2, triangle.p3 };
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
                Point4D p1 = p[outside[0].index];
                Point4D p2 = p[inside[0].index];
                Point4D p3 = p[inside[1].index];
                Point4D pi1 = CalculateIntersectionPointLine(p1, p2, plane);
                Point4D pi2 = CalculateIntersectionPointLine(p3, p1, plane);
                yield return new Triangle { p1 = p1, p2 = pi1, p3 = pi2 };
                yield break;
            }
            // If two points are clipped then emit two triangles.
            if (outside.Length == 2 && inside.Length == 1)
            {
                int first = outside[0].index;
                Point4D p1 = p[outside[0].index];
                Point4D p2 = p[outside[1].index];
                Point4D p3 = p[inside[0].index];
                Point4D p2i = CalculateIntersectionPointLine(p2, p3, plane);
                Point4D p3i = CalculateIntersectionPointLine(p3, p1, plane);
                yield return new Triangle { p1 = p1, p2 = p2, p3 = p2i };
                yield return new Triangle { p1 = p2i, p2 = p3i, p3 = p1 };
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