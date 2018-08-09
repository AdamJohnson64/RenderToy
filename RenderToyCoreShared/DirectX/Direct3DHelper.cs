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
    public struct Direct3DHelper
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
                Position = new Vector3F((float)i.Position.X, (float)i.Position.Y, (float)i.Position.Z)
            });
            return verticesout.ToArray();
        }
        public static XYZNorDiffuseTex1[] ConvertToXYZNorDiffuseTex1(IPrimitive primitive)
        {
            var verticesin = PrimitiveAssembly.CreateTrianglesDX(primitive);
            var verticesout = verticesin.Select(i => new XYZNorDiffuseTex1
            {
                Position = new Vector3F((float)i.Position.X, (float)i.Position.Y, (float)i.Position.Z),
                Normal = new Vector3F((float)i.Normal.X, (float)i.Normal.Y, (float)i.Normal.Z),
                Diffuse = i.Diffuse,
                TexCoord = new Vector2F((float)i.TexCoord.X, (float)i.TexCoord.Y),
                Tangent = new Vector3F((float)i.Tangent.X, (float)i.Tangent.Y, (float)i.Tangent.Z),
                Bitangent = new Vector3F((float)i.Bitangent.X, (float)i.Bitangent.Y, (float)i.Bitangent.Z),
            });
            return verticesout.ToArray();
        }
    }
}