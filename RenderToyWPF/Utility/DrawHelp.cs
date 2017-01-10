////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static class DrawHelp
    {
        #region - Section : Low-Level Primitives -
        /// <summary>
        /// Generic viewport line drawing function draws a line in viewport space.
        /// Viewport lines are expressed (-1, -1) -> (+1, +1).
        /// </summary>
        /// <param name="p1">The viewport space starting position.</param>
        /// <param name="p2">The viewport space ending position.</param>
        public delegate void fnDrawLineViewport(Point p1, Point p2);
        /// <summary>
        /// Clip, prepare and draw a line in 3D world space.
        /// </summary>
        /// <param name="line">The viewport line rendering function.</param>
        /// <param name="mvp">The model-view-projection transform.</param>
        /// <param name="p1">The world space starting position.</param>
        /// <param name="p2">The world space ending position.</param>
        public static void DrawLineWorld(DrawHelp.fnDrawLineViewport line, Matrix3D mvp, Point4D p1, Point4D p2)
        {
            if (!TransformAndClipLine(ref p1, ref p2, mvp)) return;
            // Perform homogeneous divide and draw the viewport space line.
            line(new Point(p1.X / p1.W, p1.Y / p1.W), new Point(p2.X / p2.W, p2.Y / p2.W));
        }
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
        private static bool ClipLine3D(ref Point4D p1, ref Point4D p2)
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
        private static bool ClipLine3D(ref Point4D p1, ref Point4D p2, Point4D plane)
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
            double lambda = -MathHelp.Dot(plane, line_org) / MathHelp.Dot(plane, line_dir);
            // If the intersection lies in the line segment then clip.
            if (lambda > 0 && lambda < 1)
            {
                // If P1 was behind the plane them move it to the intersection point.
                if (side_p1 <= 0) p1 = line_org + MathHelp.Scale(line_dir, lambda);
                // If P2 was behind the plane then move it to the intersection point.
                if (side_p2 <= 0) p2 = line_org + MathHelp.Scale(line_dir, lambda);
            }
            return true;
        }
        private static double IntersectLine3D(Point4D p1, Point4D p2, Point4D plane)
        {
            // Compute the intersection with the clip plane.
            return -MathHelp.Dot(plane, p1) / MathHelp.Dot(plane, p2 - p1);
        }
        private static Point4D IntersectLine3DPoint(Point4D p1, Point4D p2, Point4D plane)
        {
            return p1 + MathHelp.Scale(p2 - p1, IntersectLine3D(p1, p2, plane));
        }
        public struct Triangle
        {
            public Point4D p1, p2, p3;
        }
        public static IEnumerable<Triangle> ClipTriangle3D(Triangle tri)
        {
            return ClipTriangle3D(new Triangle[] { tri });
        }
        public static IEnumerable<Triangle> ClipTriangle3D(IEnumerable<Triangle> triangles)
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
        public static IEnumerable<Triangle> ClipTriangle3D(Triangle triangle, Point4D plane)
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
                Point4D pi1 = IntersectLine3DPoint(p1, p2, plane);
                Point4D pi2 = IntersectLine3DPoint(p3, p1, plane);
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
                Point4D p2i = IntersectLine3DPoint(p2, p3, plane);
                Point4D p3i = IntersectLine3DPoint(p3, p1, plane);
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
        #region - Section : Higher Order Primitives (Points) -
        /// <summary>
        /// Generic point drawing function draws a point in 3D space.
        /// </summary>
        /// <param name="p">The world space point.</param>
        public delegate void fnDrawPointWorld(Point3D p);
        /// <summary>
        /// Draw a uv parametric surface as a series of points.
        /// </summary>
        /// <param name="point">The point rendering function.</param>
        /// <param name="shape">The uv parametric surface to draw.</param>
        public static void DrawParametricUV(fnDrawPointWorld point, IParametricUV shape)
        {
            int USEGMENTS = 20;
            int VSEGMENTS = 20;
            // Simply move some number of steps across u and v and draw the points in space.
            for (int u = 0; u < USEGMENTS; ++u)
            {
                for (int v = 0; v < VSEGMENTS; ++v)
                {
                    // Determine the point and draw it; easy.
                    point(shape.GetPointUV((double)u / USEGMENTS, (double)v / VSEGMENTS));
                }
            }
        }
        #endregion
        #region - Section : Higher Order Primitives (Wireframe) -
        /// <summary>
        /// Generic line drawing function draws a line in 3D space.
        /// </summary>
        /// <param name="p1">The world space starting position.</param>
        /// <param name="p2">The world space ending position.</param>
        public delegate void fnDrawLineWorld(Point3D p1, Point3D p2);
        /// <summary>
        /// Draw a wireframe representing a clip space.
        /// This specialized function will render a visualization of a clip space.
        /// The four homogeneous clip corners (-1, -1, 0, 1) -> (+1, +1, 0, 1) are unprojected back into 3-space and rendered as a warped box.
        /// In the case of the perspective projection this will yield the clip space frustum.
        /// </summary>
        /// <param name="line">The world space line rendering function.</param>
        /// <param name="clipspace">The inverse MVP matrix for the clip space.</param>
        public static void DrawClipSpace(fnDrawLineWorld line, Matrix3D clipspace)
        {
            {
                // Transform the edges of homogeneous space into 3-space.
                Point3D[] points = new Point3D[8];
                for (int z = 0; z < 2; ++z)
                {
                    for (int y = 0; y < 2; ++y)
                    {
                        for (int x = 0; x < 2; ++x)
                        {
                            Point4D p = clipspace.Transform(new Point4D(-1 + x * 2, -1 + y * 2, z, 1));
                            // Homogeneous divide puts us back in real space.
                            points[x + y * 2 + z * 4] = new Point3D(p.X / p.W, p.Y / p.W, p.Z / p.W);
                        }
                    }
                }
                // Draw the projection corners (the z-expanse lines from the four corners of the viewport).
                line(points[0], points[4]);
                line(points[1], points[5]);
                line(points[2], points[6]);
                line(points[3], points[7]);
            }
            {
                // Draw several depth viewport frames at constant z spacing.
                for (int z = 0; z <= 10; ++z)
                {
                    Point3D[] frame = new Point3D[4];
                    for (int y = 0; y < 2; ++y)
                    {
                        for (int x = 0; x < 2; ++x)
                        {
                            Point4D p = clipspace.Transform(new Point4D(-1 + x * 2, -1 + y * 2, z / 10.0, 1));
                            // Homogeneous divide puts us back in real space.
                            frame[x + y * 2] = new Point3D(p.X / p.W, p.Y / p.W, p.Z / p.W);
                        }
                    }
                    line(frame[0], frame[1]);
                    line(frame[0], frame[2]);
                    line(frame[1], frame[3]);
                    line(frame[2], frame[3]);
                }
            }
        }
        /// <summary>
        /// Render a wireframe for a parametric UV surface.
        /// </summary>
        /// <param name="line">The world space line rendering function.</param>
        /// <param name="shape">The parametric surface to render.</param>
        public static void DrawParametricUV(fnDrawLineWorld line, IParametricUV shape)
        {
            int USEGMENTS = 10;
            int VSEGMENTS = 20;
            for (int u = 0; u <= USEGMENTS; ++u)
            {
                for (int v = 0; v < VSEGMENTS; ++v)
                {
                    // Draw U Lines.
                    {
                        Point3D p3u1 = shape.GetPointUV((v + 0.0) / VSEGMENTS, (u + 0.0) / USEGMENTS);
                        Point3D p3u2 = shape.GetPointUV((v + 1.0) / VSEGMENTS, (u + 0.0) / USEGMENTS);
                        line(p3u1, p3u2);
                    }
                    // Draw V Lines.
                    {
                        Point3D p3u1 = shape.GetPointUV((u + 0.0) / USEGMENTS, (v + 0.0) / VSEGMENTS);
                        Point3D p3u2 = shape.GetPointUV((u + 0.0) / USEGMENTS, (v + 1.0) / VSEGMENTS);
                        line(p3u1, p3u2);
                    }
                }
            }
        }
        /// <summary>
        /// Render a wireframe for an XZ plane.
        /// </summary>
        /// <param name="line">The world space line rendering function.</param>
        public static void DrawPlane(fnDrawLineWorld line)
        {
            for (int i = 0; i <= 20; ++i)
            {
                // Draw an X line.
                float z = -10.0f + i;
                line(new Point3D(-10, 0, z), new Point3D(10, 0, z));
                // Draw a Z line.
                float x = -10.0f + i;
                line(new Point3D(x, 0, -10), new Point3D(x, 0, 10));
            }
        }
        /// <summary>
        /// TEAPOTS! :)
        /// </summary>
        /// <param name="line">he world space line rendering function</param>
        public static void DrawTeapot(DrawHelp.fnDrawLineWorld line)
        {
            Point3D[] vtx = new Point3D[] {
                new Point3D(0.2000,  0.0000, 2.70000), new Point3D(0.2000, -0.1120, 2.70000),
                new Point3D(1.3375,  0.0000, 2.53125), new Point3D(1.3375, -0.7490, 2.53125),
                new Point3D(0.7490, -1.3375, 2.53125), new Point3D(0.0000, -1.3375, 2.53125),
                new Point3D(1.4375,  0.0000, 2.53125), new Point3D(1.4375, -0.8050, 2.53125),
                new Point3D(0.1120, -0.2000, 2.70000), new Point3D(0.0000, -0.2000, 2.70000),
                new Point3D(0.8050, -1.4375, 2.53125), new Point3D(0.0000, -1.4375, 2.53125),
                new Point3D(1.5000,  0.0000, 2.40000), new Point3D(1.5000, -0.8400, 2.40000),
                new Point3D(0.8400, -1.5000, 2.40000), new Point3D(0.0000, -1.5000, 2.40000),
                new Point3D(1.7500,  0.0000, 1.87500), new Point3D(1.7500, -0.9800, 1.87500),
                new Point3D(0.9800, -1.7500, 1.87500), new Point3D(0.0000, -1.7500, 1.87500),
                new Point3D(2.0000,  0.0000, 1.35000), new Point3D(2.0000, -1.1200, 1.35000),
                new Point3D(1.1200, -2.0000, 1.35000), new Point3D(0.0000, -2.0000, 1.35000),
                new Point3D(2.0000,  0.0000, 0.90000), new Point3D(2.0000, -1.1200, 0.90000),
                new Point3D(1.1200, -2.0000, 0.90000), new Point3D(0.0000, -2.0000, 0.90000),
                new Point3D(-2.0000,  0.0000, 0.90000), new Point3D(2.0000,  0.0000, 0.45000),
                new Point3D(2.0000, -1.1200, 0.45000), new Point3D(1.1200, -2.0000, 0.45000),
                new Point3D(0.0000, -2.0000, 0.45000), new Point3D(1.5000,  0.0000, 0.22500),
                new Point3D(1.5000, -0.8400, 0.22500), new Point3D(0.8400, -1.5000, 0.22500),
                new Point3D(0.0000, -1.5000, 0.22500), new Point3D(1.5000,  0.0000, 0.15000),
                new Point3D(1.5000, -0.8400, 0.15000), new Point3D(0.8400, -1.5000, 0.15000),
                new Point3D(0.0000, -1.5000, 0.15000), new Point3D(-1.6000,  0.0000, 2.02500),
                new Point3D(-1.6000, -0.3000, 2.02500), new Point3D(-1.5000, -0.3000, 2.25000),
                new Point3D(-1.5000,  0.0000, 2.25000), new Point3D(-2.3000,  0.0000, 2.02500),
                new Point3D(-2.3000, -0.3000, 2.02500), new Point3D(-2.5000, -0.3000, 2.25000),
                new Point3D(-2.5000,  0.0000, 2.25000), new Point3D(-2.7000,  0.0000, 2.02500),
                new Point3D(-2.7000, -0.3000, 2.02500), new Point3D(-3.0000, -0.3000, 2.25000),
                new Point3D(-3.0000,  0.0000, 2.25000), new Point3D(-2.7000,  0.0000, 1.80000),
                new Point3D(-2.7000, -0.3000, 1.80000), new Point3D(-3.0000, -0.3000, 1.80000),
                new Point3D(-3.0000,  0.0000, 1.80000), new Point3D(-2.7000,  0.0000, 1.57500),
                new Point3D(-2.7000, -0.3000, 1.57500), new Point3D(-3.0000, -0.3000, 1.35000),
                new Point3D(-3.0000,  0.0000, 1.35000), new Point3D(-2.5000,  0.0000, 1.12500),
                new Point3D(-2.5000, -0.3000, 1.12500), new Point3D(-2.6500, -0.3000, 0.93750),
                new Point3D(-2.6500,  0.0000, 0.93750), new Point3D(-2.0000, -0.3000, 0.90000),
                new Point3D(-1.9000, -0.3000, 0.60000), new Point3D(-1.9000,  0.0000, 0.60000),
                new Point3D(1.7000,  0.0000, 1.42500), new Point3D(1.7000, -0.6600, 1.42500),
                new Point3D(1.7000, -0.6600, 0.60000), new Point3D(1.7000,  0.0000, 0.60000),
                new Point3D(2.6000,  0.0000, 1.42500), new Point3D(2.6000, -0.6600, 1.42500),
                new Point3D(3.1000, -0.6600, 0.82500), new Point3D(3.1000,  0.0000, 0.82500),
                new Point3D(2.3000,  0.0000, 2.10000), new Point3D(2.3000, -0.2500, 2.10000),
                new Point3D(2.4000, -0.2500, 2.02500), new Point3D(2.4000,  0.0000, 2.02500),
                new Point3D(2.7000,  0.0000, 2.40000), new Point3D(2.7000, -0.2500, 2.40000),
                new Point3D(3.3000, -0.2500, 2.40000), new Point3D(3.3000,  0.0000, 2.40000),
                new Point3D(2.8000,  0.0000, 2.47500), new Point3D(2.8000, -0.2500, 2.47500),
                new Point3D(3.5250, -0.2500, 2.49375), new Point3D(3.5250,  0.0000, 2.49375),
                new Point3D(2.9000,  0.0000, 2.47500), new Point3D(2.9000, -0.1500, 2.47500),
                new Point3D(3.4500, -0.1500, 2.51250), new Point3D(3.4500,  0.0000, 2.51250),
                new Point3D(2.8000,  0.0000, 2.40000), new Point3D(2.8000, -0.1500, 2.40000),
                new Point3D(3.2000, -0.1500, 2.40000), new Point3D(3.2000,  0.0000, 2.40000),
                new Point3D(0.0000,  0.0000, 3.15000), new Point3D(0.8000,  0.0000, 3.15000),
                new Point3D(0.8000, -0.4500, 3.15000), new Point3D(0.4500, -0.8000, 3.15000),
                new Point3D(0.0000, -0.8000, 3.15000), new Point3D(0.0000,  0.0000, 2.85000),
                new Point3D(1.4000,  0.0000, 2.40000), new Point3D(1.4000, -0.7840, 2.40000),
                new Point3D(0.7840, -1.4000, 2.40000), new Point3D(0.0000, -1.4000, 2.40000),
                new Point3D(0.4000,  0.0000, 2.55000), new Point3D(0.4000, -0.2240, 2.55000),
                new Point3D(0.2240, -0.4000, 2.55000), new Point3D(0.0000, -0.4000, 2.55000),
                new Point3D(1.3000,  0.0000, 2.55000), new Point3D(1.3000, -0.7280, 2.55000),
                new Point3D(0.7280, -1.3000, 2.55000), new Point3D(0.0000, -1.3000, 2.55000),
                new Point3D(1.3000,  0.0000, 2.40000), new Point3D(1.3000, -0.7280, 2.40000),
                new Point3D(0.7280, -1.3000, 2.40000), new Point3D(0.0000, -1.3000, 2.40000) };
            int[] idx = new int[] {
                //Rim
                //102, 103, 104, 105, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                //Body
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                //Lid
                //96, 96, 96, 96, 97, 98, 99, 100, 101, 101, 101, 101, 0, 1, 2, 3,
                //0, 1, 2, 3, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                //Handle
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 28, 65, 66, 67,
                //Spout
                68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
            };
            int[] flips = new int[] { /*3, */3, 3, /*3, 3,*/ 2, 2, 2, 2 };
            for (int patch = 0; patch < idx.Length / 16; ++patch)
            {
                Point3D[] hull = new Point3D[16];
                for (int i = 0; i < 16; ++i)
                {
                    hull[i] = vtx[idx[patch * 16 + i]];
                    double s = hull[i].Y;
                    hull[i].Y = hull[i].Z;
                    hull[i].Z = s;
                    // There's something odd about this teapot but I can't find any authoritive hull definition online.
                    // Draw some lines for the hull points.
                    //line(hull[i], MathHelp.Add(hull[i], new Point3D(0, 0.5, 0)));
                }
                DrawHelp.DrawParametricUV(line, new BezierPatch(hull));
                if ((flips[patch] & 1) == 1) // Flip in X
                {
                    Point3D[] hull2 = new Point3D[16];
                    for (int i = 0; i < 16; ++i)
                    {
                        hull2[i] = hull[i];
                        hull2[i].X = -hull2[i].X;
                    }
                    DrawHelp.DrawParametricUV(line, new BezierPatch(hull2));
                }
                if ((flips[patch] & 2) == 2) // Flip in Z
                {
                    Point3D[] hull2 = new Point3D[16];
                    for (int i = 0; i < 16; ++i)
                    {
                        hull2[i] = hull[i];
                        hull2[i].Z = -hull2[i].Z;
                    }
                    DrawHelp.DrawParametricUV(line, new BezierPatch(hull2));
                }
                if ((flips[patch] & 3) == 3) // Flip in X and Z
                {
                    Point3D[] hull2 = new Point3D[16];
                    for (int i = 0; i < 16; ++i)
                    {
                        hull2[i] = hull[i];
                        hull2[i].X = -hull2[i].X;
                        hull2[i].Z = -hull2[i].Z;
                    }
                    DrawHelp.DrawParametricUV(line, new BezierPatch(hull2));
                }
            }
        }
        #endregion
    }
}
