////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

namespace RenderToy.Meshes
{
    public struct XYZ
    {
        public float Xp, Yp, Zp;
    };
    public struct XYZDiffuse
    {
        public float X, Y, Z;
        public uint Diffuse;
    };
    public struct XYZWDiffuse
    {
        public float X, Y, Z, W;
        public uint Diffuse;
    };
    public struct XYZNorDiffuseTex1
    {
        public float Xp, Yp, Zp;
        public float Xn, Yn, Zn;
        public uint Diffuse;
        public float U, V;
        public float Tx, Ty, Tz;
        public float Bx, By, Bz;
    };
}