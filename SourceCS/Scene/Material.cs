////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;

namespace RenderToy
{
    public interface IMaterial
    {
        Point4D MaterialCompute(Point3D origin, Vector3D direction, double lambda);
    }
    public static class Materials
    {
        public static Point4D Empty = new Point4D(0, 0, 0, 0);
        public static Point4D Percent0 = new Point4D(0, 0, 0, 0);
        public static Point4D Percent50 = new Point4D(0.5, 0.5, 0.5, 0.5);
        public static Point4D Percent100 = new Point4D(1, 1, 1, 1);
        public static Point4D Black = new Point4D(0, 0, 0, 1);
        public static Point4D Red = new Point4D(1, 0, 0, 1);
        public static Point4D Green = new Point4D(0, 0.5, 0, 1);
        public static Point4D Blue = new Point4D(0, 0, 1, 1);
        public static Point4D Yellow = new Point4D(1, 1, 0, 1);
        public static Point4D Magenta = new Point4D(1, 0, 1, 1);
        public static Point4D Cyan = new Point4D(0, 1, 1, 1);
        public static Point4D White = new Point4D(1, 1, 1, 1);
        public static Point4D DarkGray = new Point4D(0.25, 0.25, 0.25, 1);
        public static Point4D LightGray = new Point4D(0.75, 0.75, 0.75, 1);
        public static MaterialCommon PlasticRed = new MaterialCommon(Empty, Red, White, Percent50, Percent0, 1);
        public static MaterialCommon PlasticGreen = new MaterialCommon(Empty, Green, White, Percent50, Percent0, 1);
        public static MaterialCommon PlasticBlue = new MaterialCommon(Empty, Blue, White, Percent50, Percent0, 1);
        public static MaterialCommon PlasticYellow = new MaterialCommon(Empty, Yellow, White, Percent50, Percent0, 1);
        public static MaterialCommon PlasticMagenta = new MaterialCommon(Empty, Magenta, White, Percent50, Percent0, 1);
        public static MaterialCommon PlasticCyan = new MaterialCommon(Empty, Cyan, White, Percent50, Percent0, 1);
        public static MaterialCommon Glass = new MaterialCommon(Empty, Empty, White, Percent0, Percent100, 1.5);
    }
    public class CheckerboardMaterial : IMaterial
    {
        public CheckerboardMaterial(Point4D color1, Point4D color2)
        {
            this.Color1 = color1;
            this.Color2 = color2;
        }
        public Point4D MaterialCompute(Point3D origin, Vector3D direction, double lambda)
        {
            Point3D space = origin + lambda * direction;
            double mx = Math.Round(space.X - Math.Floor(space.X));
            double mz = Math.Round(space.Z - Math.Floor(space.Z));
            double mod = (mx + mz) % 2;
            return mod < 0.5 ? Color1 : Color2;
        }
        private Point4D Color1;
        private Point4D Color2;
    }
    public class MaterialCommon : IMaterial
    {
        public MaterialCommon(Point4D ambient, Point4D diffuse, Point4D specular, Point4D reflect, Point4D refract, double ior)
        {
            Ambient = ambient;
            Diffuse = diffuse;
            Specular = specular;
            Reflect = reflect;
            Refract = refract;
            Ior = ior;
        }
        public Point4D MaterialCompute(Point3D origin, Vector3D direction, double lambda)
        {
            return Diffuse;
        }
        public readonly Point4D Ambient;
        public readonly Point4D Diffuse;
        public readonly Point4D Specular;
        public readonly Point4D Reflect;
        public readonly Point4D Refract;
        public readonly double Ior;
    }
}