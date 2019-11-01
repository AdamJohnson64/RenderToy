////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;

namespace RenderToy.Meshes
{
    public struct XYZ
    {
        public Vector3F Position;
    };
    public struct XYZDiffuse
    {
        public Vector3F Position;
        public uint Diffuse;
    };
    public struct XYZWDiffuse
    {
        public Vector4F Position;
        public uint Diffuse;
    };
    public struct XYZNorDiffuseTex1
    {
        public Vector3F Position;
        public Vector3F Normal;
        public uint Diffuse;
        public Vector2F TexCoord;
        public Vector3F Tangent;
        public Vector3F Bitangent;
    };
}