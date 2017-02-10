////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace RenderToy
{
    public partial class MathHelp
    {
        #region - Section : Top Level Hemisphere Sample Generators -
        public static IEnumerable<Vector3D> HemiHaltonCosineBias(int count)
        {
            return UVHalton(count).Select(x => HemisphereCosineBias(x));
        }
        public static IEnumerable<Vector3D> HemiHaltonUnbiased(int count)
        {
            return UVHalton(count).Select(x => HemisphereUnbiased(x));
        }
        public static IEnumerable<Vector3D> HemiHammerslyCosineBias(int count)
        {
            return UVHammersley(count).Select(x => HemisphereCosineBias(x));
        }
        public static IEnumerable<Vector3D> HemiHammerslyUnbiased(int count)
        {
            return UVHammersley(count).Select(x => HemisphereUnbiased(x));
        }
        public static IEnumerable<Vector3D> HemiRandomCosineBias(int count)
        {
            return UVRandom(count).Select(x => HemisphereCosineBias(x));
        }
        public static IEnumerable<Vector3D> HemiRandomUnbiased(int count)
        {
            return UVRandom(count).Select(x => HemisphereUnbiased(x));
        }
        public static IEnumerable<Vector3D> HemiRandomCube(int count)
        {
            Random r = new Random(123456789);
            while (count > 0)
            {
                // Generate XYZ coordinates inside the hemi-cube.
                double x = -1 + r.NextDouble() * 2;
                double y = -1 + r.NextDouble() * 2;
                double z = r.NextDouble();
                Vector3D v = new Vector3D(x, y, z);
                // Precompute the length of this vector.
                double l = MathHelp.Length(v);
                // Reject points that are not inside the unit sphere.
                // If you don't do this you'll get a non-uniform distribution clumped at the corners.
                if (l > 1) continue;
                // Bring the points back onto the sphere.
                yield return MathHelp.Multiply(v, 1 / l);
                --count;
            }
        }
        #endregion
        #region - Section : Sequence Generators -
        public static double SequenceHalton(int radix, int x)
        {
            double result = 0;
            int div = radix;
            while (x > 0)
            {
                int mod = x % radix;
                result += (double)mod / div;
                x = x / radix;
                div = div * radix;
            }
            return result;
        }
        static double SequenceVanDerCorput(int x)
        {
            // Van der Corput sequence (Halton sequence for base-2).
            // For each bit set in the index iterator we sum the corresponding binary fraction.
            // Bit 0 : 1.0 / 2.0 = +0.5
            // Bit 1 : 1.0 / 4.0 = +0.25
            // Bit 2 : 1.0 / 8.0 = +0.125
            // Bit 3 : 1.0 / 16.0 = +0.0625
            // * Note: This isn't intended to be FAST and there are much
            // * faster ways to achieve this with float bit-hacking.
            double result = 0;
            for (int bits = 0; bits < 24; ++bits)
            {          
                if ((x & (1 << bits)) != 0) result += 1.0 / (2 << bits);
            }
            return result;
        }
        #endregion
        #region - Section : UV Coordinate Generators -
        static IEnumerable<Point2> UVHalton(int count)
        {
            for (int x = 0; x < count; ++x)
            {
                yield return new Point2(SequenceHalton(2, x), SequenceHalton(3, x));
            }
        }
        static IEnumerable<Point2> UVHammersley(int count)
        {
            for (int x = 0; x < count; ++x)
            {
                yield return new Point2((double)x / count, SequenceVanDerCorput(x));
            }
        }
        static IEnumerable<Point2> UVRandom(int count)
        {
            Random r = new Random(123456789);
            while (count > 0)
            {
                double u = r.NextDouble();
                double v = r.NextDouble();
                yield return new Point2(u, v);
                --count;
            }
        }
        struct Point2
        {
            public Point2(double u, double v)
            {
                U = u;
                V = v;
            }
            public readonly double U;
            public readonly double V;
        }
        #endregion
        #region - Section : Hemisphere Converters -
        static Vector3D HemisphereCosineBias(Point2 uv)
        {
            double azimuth = uv.U * 2 * Math.PI;
            double altitude = Math.Acos(Math.Sqrt(uv.V));
            double cos_alt = Math.Cos(altitude);
            return new Vector3D(Math.Sin(azimuth) * cos_alt, Math.Cos(azimuth) * cos_alt, Math.Sin(altitude));
        }
        static Vector3D HemisphereUnbiased(Point2 uv)
        {
            double azimuth = uv.U * 2 * Math.PI;
            double altitude = Math.Acos(uv.V);
            double cos_alt = Math.Cos(altitude);
            return new Vector3D(Math.Sin(azimuth) * cos_alt, Math.Cos(azimuth) * cos_alt, Math.Sin(altitude));
        }
        #endregion
    }
}