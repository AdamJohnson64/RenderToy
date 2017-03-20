////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy.SceneGraph.Materials
{
    public class GenericMaterial : IMaterial
    {
        public GenericMaterial(Vector4D ambient, Vector4D diffuse, Vector4D specular, Vector4D reflect, Vector4D refract, double ior)
        {
            Ambient = ambient;
            Diffuse = diffuse;
            Specular = specular;
            Reflect = reflect;
            Refract = refract;
            Ior = ior;
        }
        public readonly Vector4D Ambient;
        public readonly Vector4D Diffuse;
        public readonly Vector4D Specular;
        public readonly Vector4D Reflect;
        public readonly Vector4D Refract;
        public readonly double Ior;
    }
}