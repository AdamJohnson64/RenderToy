////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;

namespace RenderToy
{
    public interface IMaterial
    {
        Vector4D MaterialCompute(Vector3D origin, Vector3D direction, double lambda);
    }
    public static class Materials
    {
        public static Vector4D Empty = new Vector4D(0, 0, 0, 0);
        public static Vector4D Percent0 = new Vector4D(0, 0, 0, 0);
        public static Vector4D Percent50 = new Vector4D(0.5, 0.5, 0.5, 0.5);
        public static Vector4D Percent100 = new Vector4D(1, 1, 1, 1);
        public static Vector4D Black = new Vector4D(0, 0, 0, 1);
        public static Vector4D Red = new Vector4D(1, 0, 0, 1);
        public static Vector4D Green = new Vector4D(0, 0.5, 0, 1);
        public static Vector4D Blue = new Vector4D(0, 0, 1, 1);
        public static Vector4D Yellow = new Vector4D(1, 1, 0, 1);
        public static Vector4D Magenta = new Vector4D(1, 0, 1, 1);
        public static Vector4D Cyan = new Vector4D(0, 1, 1, 1);
        public static Vector4D White = new Vector4D(1, 1, 1, 1);
        public static Vector4D DarkGray = new Vector4D(0.25, 0.25, 0.25, 1);
        public static Vector4D LightGray = new Vector4D(0.75, 0.75, 0.75, 1);
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
        public CheckerboardMaterial(Vector4D color1, Vector4D color2)
        {
            this.Color1 = color1;
            this.Color2 = color2;
        }
        public Vector4D MaterialCompute(Vector3D origin, Vector3D direction, double lambda)
        {
            Vector3D space = origin + lambda * direction;
            double mx = Math.Round(space.X - Math.Floor(space.X));
            double mz = Math.Round(space.Z - Math.Floor(space.Z));
            double mod = (mx + mz) % 2;
            return mod < 0.5 ? Color1 : Color2;
        }
        private Vector4D Color1;
        private Vector4D Color2;
    }
    public class MaterialCommon : IMaterial
    {
        public MaterialCommon(Vector4D ambient, Vector4D diffuse, Vector4D specular, Vector4D reflect, Vector4D refract, double ior)
        {
            Ambient = ambient;
            Diffuse = diffuse;
            Specular = specular;
            Reflect = reflect;
            Refract = refract;
            Ior = ior;
        }
        public Vector4D MaterialCompute(Vector3D origin, Vector3D direction, double lambda)
        {
            return Diffuse;
        }
        public readonly Vector4D Ambient;
        public readonly Vector4D Diffuse;
        public readonly Vector4D Specular;
        public readonly Vector4D Reflect;
        public readonly Vector4D Refract;
        public readonly double Ior;
    }
}