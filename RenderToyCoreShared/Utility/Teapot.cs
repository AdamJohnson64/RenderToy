////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy
{
    public static class DrawHelp
    {
        /*
        #region - Section : Higher Order Primitives (Wireframe) -
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
                Vector3D[] points = new Vector3D[8];
                for (int z = 0; z < 2; ++z)
                {
                    for (int y = 0; y < 2; ++y)
                    {
                        for (int x = 0; x < 2; ++x)
                        {
                            Vector4D p = clipspace.Transform(new Vector4D(-1 + x * 2, -1 + y * 2, z, 1));
                            // Homogeneous divide puts us back in real space.
                            points[x + y * 2 + z * 4] = new Vector3D(p.X / p.W, p.Y / p.W, p.Z / p.W);
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
                    Vector3D[] frame = new Vector3D[4];
                    for (int y = 0; y < 2; ++y)
                    {
                        for (int x = 0; x < 2; ++x)
                        {
                            Vector4D p = clipspace.Transform(new Vector4D(-1 + x * 2, -1 + y * 2, z / 10.0, 1));
                            // Homogeneous divide puts us back in real space.
                            frame[x + y * 2] = new Vector3D(p.X / p.W, p.Y / p.W, p.Z / p.W);
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
        /// TEAPOTS! :)
        /// </summary>
        /// <param name="line">he world space line rendering function</param>
        public static void DrawTeapot(fnDrawLineWorld line)
        {
            Vector3D[] vtx = new Vector3D[] {
                new Vector3D(0.2000,  0.0000, 2.70000), new Vector3D(0.2000, -0.1120, 2.70000),
                new Vector3D(1.3375,  0.0000, 2.53125), new Vector3D(1.3375, -0.7490, 2.53125),
                new Vector3D(0.7490, -1.3375, 2.53125), new Vector3D(0.0000, -1.3375, 2.53125),
                new Vector3D(1.4375,  0.0000, 2.53125), new Vector3D(1.4375, -0.8050, 2.53125),
                new Vector3D(0.1120, -0.2000, 2.70000), new Vector3D(0.0000, -0.2000, 2.70000),
                new Vector3D(0.8050, -1.4375, 2.53125), new Vector3D(0.0000, -1.4375, 2.53125),
                new Vector3D(1.5000,  0.0000, 2.40000), new Vector3D(1.5000, -0.8400, 2.40000),
                new Vector3D(0.8400, -1.5000, 2.40000), new Vector3D(0.0000, -1.5000, 2.40000),
                new Vector3D(1.7500,  0.0000, 1.87500), new Vector3D(1.7500, -0.9800, 1.87500),
                new Vector3D(0.9800, -1.7500, 1.87500), new Vector3D(0.0000, -1.7500, 1.87500),
                new Vector3D(2.0000,  0.0000, 1.35000), new Vector3D(2.0000, -1.1200, 1.35000),
                new Vector3D(1.1200, -2.0000, 1.35000), new Vector3D(0.0000, -2.0000, 1.35000),
                new Vector3D(2.0000,  0.0000, 0.90000), new Vector3D(2.0000, -1.1200, 0.90000),
                new Vector3D(1.1200, -2.0000, 0.90000), new Vector3D(0.0000, -2.0000, 0.90000),
                new Vector3D(-2.0000,  0.0000, 0.90000), new Vector3D(2.0000,  0.0000, 0.45000),
                new Vector3D(2.0000, -1.1200, 0.45000), new Vector3D(1.1200, -2.0000, 0.45000),
                new Vector3D(0.0000, -2.0000, 0.45000), new Vector3D(1.5000,  0.0000, 0.22500),
                new Vector3D(1.5000, -0.8400, 0.22500), new Vector3D(0.8400, -1.5000, 0.22500),
                new Vector3D(0.0000, -1.5000, 0.22500), new Vector3D(1.5000,  0.0000, 0.15000),
                new Vector3D(1.5000, -0.8400, 0.15000), new Vector3D(0.8400, -1.5000, 0.15000),
                new Vector3D(0.0000, -1.5000, 0.15000), new Vector3D(-1.6000,  0.0000, 2.02500),
                new Vector3D(-1.6000, -0.3000, 2.02500), new Vector3D(-1.5000, -0.3000, 2.25000),
                new Vector3D(-1.5000,  0.0000, 2.25000), new Vector3D(-2.3000,  0.0000, 2.02500),
                new Vector3D(-2.3000, -0.3000, 2.02500), new Vector3D(-2.5000, -0.3000, 2.25000),
                new Vector3D(-2.5000,  0.0000, 2.25000), new Vector3D(-2.7000,  0.0000, 2.02500),
                new Vector3D(-2.7000, -0.3000, 2.02500), new Vector3D(-3.0000, -0.3000, 2.25000),
                new Vector3D(-3.0000,  0.0000, 2.25000), new Vector3D(-2.7000,  0.0000, 1.80000),
                new Vector3D(-2.7000, -0.3000, 1.80000), new Vector3D(-3.0000, -0.3000, 1.80000),
                new Vector3D(-3.0000,  0.0000, 1.80000), new Vector3D(-2.7000,  0.0000, 1.57500),
                new Vector3D(-2.7000, -0.3000, 1.57500), new Vector3D(-3.0000, -0.3000, 1.35000),
                new Vector3D(-3.0000,  0.0000, 1.35000), new Vector3D(-2.5000,  0.0000, 1.12500),
                new Vector3D(-2.5000, -0.3000, 1.12500), new Vector3D(-2.6500, -0.3000, 0.93750),
                new Vector3D(-2.6500,  0.0000, 0.93750), new Vector3D(-2.0000, -0.3000, 0.90000),
                new Vector3D(-1.9000, -0.3000, 0.60000), new Vector3D(-1.9000,  0.0000, 0.60000),
                new Vector3D(1.7000,  0.0000, 1.42500), new Vector3D(1.7000, -0.6600, 1.42500),
                new Vector3D(1.7000, -0.6600, 0.60000), new Vector3D(1.7000,  0.0000, 0.60000),
                new Vector3D(2.6000,  0.0000, 1.42500), new Vector3D(2.6000, -0.6600, 1.42500),
                new Vector3D(3.1000, -0.6600, 0.82500), new Vector3D(3.1000,  0.0000, 0.82500),
                new Vector3D(2.3000,  0.0000, 2.10000), new Vector3D(2.3000, -0.2500, 2.10000),
                new Vector3D(2.4000, -0.2500, 2.02500), new Vector3D(2.4000,  0.0000, 2.02500),
                new Vector3D(2.7000,  0.0000, 2.40000), new Vector3D(2.7000, -0.2500, 2.40000),
                new Vector3D(3.3000, -0.2500, 2.40000), new Vector3D(3.3000,  0.0000, 2.40000),
                new Vector3D(2.8000,  0.0000, 2.47500), new Vector3D(2.8000, -0.2500, 2.47500),
                new Vector3D(3.5250, -0.2500, 2.49375), new Vector3D(3.5250,  0.0000, 2.49375),
                new Vector3D(2.9000,  0.0000, 2.47500), new Vector3D(2.9000, -0.1500, 2.47500),
                new Vector3D(3.4500, -0.1500, 2.51250), new Vector3D(3.4500,  0.0000, 2.51250),
                new Vector3D(2.8000,  0.0000, 2.40000), new Vector3D(2.8000, -0.1500, 2.40000),
                new Vector3D(3.2000, -0.1500, 2.40000), new Vector3D(3.2000,  0.0000, 2.40000),
                new Vector3D(0.0000,  0.0000, 3.15000), new Vector3D(0.8000,  0.0000, 3.15000),
                new Vector3D(0.8000, -0.4500, 3.15000), new Vector3D(0.4500, -0.8000, 3.15000),
                new Vector3D(0.0000, -0.8000, 3.15000), new Vector3D(0.0000,  0.0000, 2.85000),
                new Vector3D(1.4000,  0.0000, 2.40000), new Vector3D(1.4000, -0.7840, 2.40000),
                new Vector3D(0.7840, -1.4000, 2.40000), new Vector3D(0.0000, -1.4000, 2.40000),
                new Vector3D(0.4000,  0.0000, 2.55000), new Vector3D(0.4000, -0.2240, 2.55000),
                new Vector3D(0.2240, -0.4000, 2.55000), new Vector3D(0.0000, -0.4000, 2.55000),
                new Vector3D(1.3000,  0.0000, 2.55000), new Vector3D(1.3000, -0.7280, 2.55000),
                new Vector3D(0.7280, -1.3000, 2.55000), new Vector3D(0.0000, -1.3000, 2.55000),
                new Vector3D(1.3000,  0.0000, 2.40000), new Vector3D(1.3000, -0.7280, 2.40000),
                new Vector3D(0.7280, -1.3000, 2.40000), new Vector3D(0.0000, -1.3000, 2.40000) };
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
            //int[] flips = new int[] { 3, 3, 3, 3, 3, 2, 2, 2, 2 };
            int[] flips = new int[] { 3, 3, 2, 2, 2, 2 };
            for (int patch = 0; patch < idx.Length / 16; ++patch)
            {
                Vector3D[] hull = new Vector3D[16];
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
                DrawHelp.DrawParametricUV(line, new BezierPatch(hull).GetPointUV);
                if ((flips[patch] & 1) == 1) // Flip in X
                {
                    Vector3D[] hull2 = new Vector3D[16];
                    for (int i = 0; i < 16; ++i)
                    {
                        hull2[i] = hull[i];
                        hull2[i].X = -hull2[i].X;
                    }
                    DrawHelp.DrawParametricUV(line, new BezierPatch(hull2).GetPointUV);
                }
                if ((flips[patch] & 2) == 2) // Flip in Z
                {
                    Vector3D[] hull2 = new Vector3D[16];
                    for (int i = 0; i < 16; ++i)
                    {
                        hull2[i] = hull[i];
                        hull2[i].Z = -hull2[i].Z;
                    }
                    DrawHelp.DrawParametricUV(line, new BezierPatch(hull2).GetPointUV);
                }
                if ((flips[patch] & 3) == 3) // Flip in X and Z
                {
                    Vector3D[] hull2 = new Vector3D[16];
                    for (int i = 0; i < 16; ++i)
                    {
                        hull2[i] = hull[i];
                        hull2[i].X = -hull2[i].X;
                        hull2[i].Z = -hull2[i].Z;
                    }
                    DrawHelp.DrawParametricUV(line, new BezierPatch(hull2).GetPointUV);
                }
            }
        }
        #endregion
        */
    }
}
