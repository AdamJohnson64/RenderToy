////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static class DrawHelp
    {
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
        /// <summary>
        /// Draw a uvw parametric volume as a series of points.
        /// </summary>
        /// <param name="point">The point rendering function.</param>
        /// <param name="shape">The uvw parametric volume to draw.</param>
        public static void DrawParametricUVW(fnDrawPointWorld point, IParametricUVW shape)
        {
            int USEGMENTS = 20;
            int VSEGMENTS = 20;
            int WSEGMENTS = 20;
            // Simply move some number of steps across u and v and draw the points in space.
            for (int u = 0; u < USEGMENTS; ++u)
            {
                for (int v = 0; v < VSEGMENTS; ++v)
                {
                    for (int w = 0; w < VSEGMENTS; ++w)
                    {
                        // Determine the point and draw it; easy.
                        point(shape.GetPointUVW((double)u / USEGMENTS, (double)v / VSEGMENTS, (double)w / WSEGMENTS));
                    }
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
            int LSEGMENTS = 20;
            for (int u = 0; u <= USEGMENTS; ++u)
            {
                for (int l = 0; l < LSEGMENTS; ++l)
                {
                    // Draw U Lines.
                    {
                        Point3D p3u1 = shape.GetPointUV((u + 0.0) / USEGMENTS, (l + 0.0) / LSEGMENTS);
                        Point3D p3u2 = shape.GetPointUV((u + 0.0) / USEGMENTS, (l + 1.0) / LSEGMENTS);
                        line(p3u1, p3u2);
                    }
                    // Draw V Lines.
                    {
                        Point3D p3u1 = shape.GetPointUV((l + 0.0) / LSEGMENTS, (u + 0.0) / USEGMENTS);
                        Point3D p3u2 = shape.GetPointUV((l + 1.0) / LSEGMENTS, (u + 0.0) / USEGMENTS);
                        line(p3u1, p3u2);
                    }
                }
            }
        }
        /// <summary>
        /// Render a wireframe for a parametric UV surface.
        /// </summary>
        /// <param name="line">The world space line rendering function.</param>
        /// <param name="shape">The parametric surface to render.</param>
        public static void DrawParametricUVW(fnDrawLineWorld line, IParametricUVW shape)
        {
            int USEGMENTS = 10;
            int LSEGMENTS = 20;
            for (int u = 0; u <= USEGMENTS; ++u)
            {
                for (int v = 0; v <= USEGMENTS; ++v)
                {
                    for (int l = 0; l < LSEGMENTS; ++l)
                    {
                        // Draw UV Lines.
                        {
                            Point3D p3u1 = shape.GetPointUVW((u + 0.0) / USEGMENTS, (v + 0.0) / USEGMENTS, (l + 0.0) / LSEGMENTS);
                            Point3D p3u2 = shape.GetPointUVW((u + 0.0) / USEGMENTS, (v + 0.0) / USEGMENTS, (l + 1.0) / LSEGMENTS);
                            line(p3u1, p3u2);
                        }
                        {
                            Point3D p3u1 = shape.GetPointUVW((v + 0.0) / USEGMENTS, (u + 0.0) / USEGMENTS, (l + 0.0) / LSEGMENTS);
                            Point3D p3u2 = shape.GetPointUVW((v + 0.0) / USEGMENTS, (u + 0.0) / USEGMENTS, (l + 1.0) / LSEGMENTS);
                            line(p3u1, p3u2);
                        }
                        // Draw UW Lines.
                        {
                            Point3D p3u1 = shape.GetPointUVW((u + 0.0) / USEGMENTS, (l + 0.0) / LSEGMENTS, (v + 0.0) / USEGMENTS);
                            Point3D p3u2 = shape.GetPointUVW((u + 0.0) / USEGMENTS, (l + 1.0) / LSEGMENTS, (v + 0.0) / USEGMENTS);
                            line(p3u1, p3u2);
                        }
                        {
                            Point3D p3u1 = shape.GetPointUVW((v + 0.0) / USEGMENTS, (l + 0.0) / LSEGMENTS, (u + 0.0) / USEGMENTS);
                            Point3D p3u2 = shape.GetPointUVW((v + 0.0) / USEGMENTS, (l + 1.0) / LSEGMENTS, (u + 0.0) / USEGMENTS);
                            line(p3u1, p3u2);
                        }
                        // Draw VW Lines.
                        {
                            Point3D p3u1 = shape.GetPointUVW((l + 0.0) / LSEGMENTS, (u + 0.0) / USEGMENTS, (v + 0.0) / USEGMENTS);
                            Point3D p3u2 = shape.GetPointUVW((l + 1.0) / LSEGMENTS, (u + 0.0) / USEGMENTS, (v + 0.0) / USEGMENTS);
                            line(p3u1, p3u2);
                        }
                        {
                            Point3D p3u1 = shape.GetPointUVW((l + 0.0) / LSEGMENTS, (v + 0.0) / USEGMENTS, (u + 0.0) / USEGMENTS);
                            Point3D p3u2 = shape.GetPointUVW((l + 1.0) / LSEGMENTS, (v + 0.0) / USEGMENTS, (u + 0.0) / USEGMENTS);
                            line(p3u1, p3u2);
                        }
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
