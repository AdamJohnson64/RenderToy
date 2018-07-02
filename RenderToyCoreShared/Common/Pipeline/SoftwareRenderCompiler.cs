using Microsoft.CSharp;
using RenderToy.Math;
using System;
using System.CodeDom;
using System.CodeDom.Compiler;
using System.IO;

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
            maketype.Members.Add(CreateVertexComponent(typeof(Vector3F), VertexUsage.Position));
            maketype.Members.Add(CreateVertexComponent(typeof(Vector3F), VertexUsage.Normal));
            maketype.Members.Add(CreateVertexComponent(typeof(Vector3F), VertexUsage.Tangent));
            maketype.Members.Add(CreateVertexComponent(typeof(Vector3F), VertexUsage.Binormal));
            maketype.Members.Add(CreateVertexComponent(typeof(Vector4F), VertexUsage.Color));
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