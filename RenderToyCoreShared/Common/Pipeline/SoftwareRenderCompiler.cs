using Microsoft.CSharp;
using RenderToy.Math;
using System;
using System.CodeDom;
using System.CodeDom.Compiler;
using System.IO;

namespace RenderToy.Math
{
    struct Vector3<TYPE>
    {
        TYPE X, Y, Z;
    }
    struct Vector4<TYPE>
    {
        TYPE X, Y, Z, W;
    }
}

namespace RenderToy.PipelineModel
{
    enum VertexUsage
    {
        Position,
        Normal,
        TexCoord,
        Tangent,
        Binormal,
        Color,
    };
    class Compiler
    {
        static CodeMemberField CreateVertexComponent(Type type, VertexUsage usage)
        {
            var makemember = new CodeMemberField(type, usage.ToString());
            makemember.Attributes = MemberAttributes.Public;
            return makemember;
        }
        static CodeTypeDeclaration CreateVertexDeclaration()
        {
            var maketype = new CodeTypeDeclaration("VertexDeclaration");
            maketype.IsStruct = true;
            maketype.Members.Add(CreateVertexComponent(typeof(Vector3<float>), VertexUsage.Position));
            maketype.Members.Add(CreateVertexComponent(typeof(Vector3<float>), VertexUsage.Normal));
            maketype.Members.Add(CreateVertexComponent(typeof(Vector3<float>), VertexUsage.Tangent));
            maketype.Members.Add(CreateVertexComponent(typeof(Vector3<float>), VertexUsage.Binormal));
            maketype.Members.Add(CreateVertexComponent(typeof(Vector4<float>), VertexUsage.Color));
            return maketype;
        }
        static CodeNamespace CreateRendererNamespace()
        {
            var makens = new CodeNamespace("RenderToy.Generated");
            makens.Types.Add(CreateVertexDeclaration());
            return makens;
        }
        public static string CreateRenderer()
        {
            var provider = new CSharpCodeProvider();
            var writer = new StringWriter();
            provider.GenerateCodeFromNamespace(CreateRendererNamespace(), writer, new CodeGeneratorOptions { BlankLinesBetweenMembers = false });
            return writer.ToString();
        }
    }
}