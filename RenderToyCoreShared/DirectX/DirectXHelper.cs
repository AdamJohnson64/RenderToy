////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.PipelineModel;
using RenderToy.Primitives;
using System.Linq;

namespace RenderToy.DirectX
{
    public struct DirectXHelper
    {
        public static float[] ConvertToD3DMatrix(Matrix3D matrix)
        {
            return new float[16]
            {
                (float)matrix.M11, (float)matrix.M12, (float)matrix.M13, (float)matrix.M14,
                (float)matrix.M21, (float)matrix.M22, (float)matrix.M23, (float)matrix.M24,
                (float)matrix.M31, (float)matrix.M32, (float)matrix.M33, (float)matrix.M34,
                (float)matrix.M41, (float)matrix.M42, (float)matrix.M43, (float)matrix.M44,
            };
        }
        public static XYZ[] ConvertToXYZ(IPrimitive primitive)
        {
            var verticesin = PrimitiveAssembly.CreateTrianglesDX(primitive);
            var verticesout = verticesin.Select(i => new XYZ
            {
                Xp = (float)i.Position.X,
                Yp = (float)i.Position.Y,
                Zp = (float)i.Position.Z });
            return verticesout.ToArray();
        }
        public static XYZNorDiffuseTex1[] ConvertToXYZNorDiffuseTex1(IPrimitive primitive)
        {
            var verticesin = PrimitiveAssembly.CreateTrianglesDX(primitive);
            var verticesout = verticesin.Select(i => new XYZNorDiffuseTex1
            {
                Xp = (float)i.Position.X,
                Yp = (float)i.Position.Y,
                Zp = (float)i.Position.Z,
                Xn = (float)i.Normal.X,
                Yn = (float)i.Normal.Y,
                Zn = (float)i.Normal.Z,
                Diffuse = i.Diffuse,
                U = (float)i.TexCoord.X,
                V = (float)i.TexCoord.Y,
                Tx = (float)i.Tangent.X,
                Ty = (float)i.Tangent.Y,
                Tz = (float)i.Tangent.Z,
                Bx = (float)i.Bitangent.X,
                By = (float)i.Bitangent.Y,
                Bz = (float)i.Bitangent.Z,
            });
            return verticesout.ToArray();
        }
    }
}