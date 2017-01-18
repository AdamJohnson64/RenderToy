////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public interface IMaterial
    {
        Color MaterialCompute(Point3D origin, Vector3D direction, double lambda);
    }
    public class CheckerboardMaterial : IMaterial
    {
        public CheckerboardMaterial(Color color1, Color color2)
        {
            this.Color1 = color1;
            this.Color2 = color2;
        }
        public Color MaterialCompute(Point3D origin, Vector3D direction, double lambda)
        {
            Point3D space = origin + lambda * direction;
            int mx = (((space.X % 1) + 1) % 1) < 0.5 ? 0 : 1;
            int my = 0; // (((space.Y % 1) + 1) % 1) < 0.5 ? 0 : 1;
            int mz = (((space.Z % 1) + 1) % 1) < 0.5 ? 0 : 1;
            int mod = (mx + my + mz) % 2;
            return mod == 0 ? Color1 : Color2;
        }
        private Color Color1;
        private Color Color2;
    }
    public class ConstantColorMaterial : IMaterial
    {
        public ConstantColorMaterial(Color color)
        {
            this.Color = color;
        }
        public Color MaterialCompute(Point3D origin, Vector3D direction, double lambda)
        {
            return Color;
        }
        private Color Color;
    }
    public class GlassMaterial : IMaterial
    {
        public Color MaterialCompute(Point3D origin, Vector3D direction, double lambda)
        {
            return Colors.LightGray;
        }
    }
}