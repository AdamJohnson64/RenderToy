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
    public static class Materials
    {
        private static Color Empty = new Color { A = 0, R = 0, G = 0, B = 0 };
        private static Color Percent0 = new Color { A = 0, R = 0, G = 0, B = 0 };
        private static Color Percent50 = new Color { A = 128, R = 128, G = 128, B = 128 };
        private static Color Percent100 = new Color { A = 255, R = 255, G = 255, B = 255 };
        public static MaterialCommon PlasticRed = new MaterialCommon(Empty, Colors.Red, Colors.White, Percent50, Percent0, 1);
        public static MaterialCommon PlasticGreen = new MaterialCommon(Empty, Colors.Green, Colors.White, Percent50, Percent0, 1);
        public static MaterialCommon PlasticBlue = new MaterialCommon(Empty, Colors.Blue, Colors.White, Percent50, Percent0, 1);
        public static MaterialCommon Glass = new MaterialCommon(Empty, Empty, Colors.White, Percent0, Percent100, 1.5);
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
    public class MaterialCommon : IMaterial
    {
        public MaterialCommon(Color ambient, Color diffuse, Color specular, Color reflect, Color refract, double ior)
        {
            Ambient = ambient;
            Diffuse = diffuse;
            Specular = specular;
            Reflect = reflect;
            Refract = refract;
            Ior = ior;
        }
        public Color MaterialCompute(Point3D origin, Vector3D direction, double lambda)
        {
            return Diffuse;
        }
        public readonly Color Ambient;
        public readonly Color Diffuse;
        public readonly Color Specular;
        public readonly Color Reflect;
        public readonly Color Refract;
        public readonly double Ior;
    }
}