using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    static class DrawHelp
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
            // Transform the supplied points into projection space.
            p1 = mvp.Transform(p1);
            p2 = mvp.Transform(p2);
            // Perform homogeneous space clipping.
            if (!ClipLine3D(ref p1, ref p2)) return;
            // Perform homogeneous divide and draw the viewport space line.
            line(new Point(p1.X / p1.W, p1.Y / p1.W), new Point(p2.X / p2.W, p2.Y / p2.W));
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
        #endregion
        #region - Section : Higher Order Primitives -
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
            int VSEGMENTS = 100;
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
        #endregion
    }
}
