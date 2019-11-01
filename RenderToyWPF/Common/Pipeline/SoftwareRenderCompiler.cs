using Microsoft.CSharp;
using RenderToy.Math;
using System;
using System.CodeDom;
using System.CodeDom.Compiler;
using System.IO;

namespace RenderToy.PipelineModel
{
    enum ConstantUsage
    {
        ModelTransform,
        ViewTransform,
        ProjectionTransform,
        ModelViewProjectionTransform,
    }
    enum VertexUsage
    {
        Position,
        Normal,
        TexCoord,
        Tangent,
        Binormal,
        Color,
    };
    public class Compiler
    {
        static CodeMemberField CreateField(Type type, ConstantUsage usage)
        {
            var makemember = new CodeMemberField(type, usage.ToString());
            makemember.Attributes = MemberAttributes.Public;
            return makemember;
        }
        static CodeMemberField CreateField(Type type, VertexUsage usage)
        {
            var makemember = new CodeMemberField(type, usage.ToString());
            makemember.Attributes = MemberAttributes.Public;
            return makemember;
        }
        static CodeTypeDeclaration CreateVertexConstantDeclaration()
        {
            var maketype = new CodeTypeDeclaration("VertexConstantDeclaration");
            maketype.IsStruct = true;
            maketype.Members.Add(CreateField(typeof(Matrix3D), ConstantUsage.ModelViewProjectionTransform));
            return maketype;
        }
        static CodeTypeDeclaration CreateVertexInputDeclaration()
        {
            var maketype = new CodeTypeDeclaration("VertexInputDeclaration");
            maketype.IsStruct = true;
            maketype.Members.Add(CreateField(typeof(Vector3F), VertexUsage.Position));
            maketype.Members.Add(CreateField(typeof(Vector3F), VertexUsage.Normal));
            maketype.Members.Add(CreateField(typeof(Vector3F), VertexUsage.Tangent));
            maketype.Members.Add(CreateField(typeof(Vector3F), VertexUsage.Binormal));
            maketype.Members.Add(CreateField(typeof(Vector4F), VertexUsage.Color));
            return maketype;
        }
        static CodeTypeDeclaration CreatePixelConstantDeclaration()
        {
            var maketype = new CodeTypeDeclaration("PixelConstantDeclaration");
            maketype.IsStruct = true;
            return maketype;
        }
        static CodeTypeDeclaration CreatePixelInputDeclaration()
        {
            var maketype = new CodeTypeDeclaration("PixelInputDeclaration");
            maketype.IsStruct = true;
            maketype.Members.Add(CreateField(typeof(Vector4F), VertexUsage.Position));
            return maketype;
        }
        static CodeMemberMethod CreateVertexShader()
        {
            var method = new CodeMemberMethod();
            method.Name = "VertexShader";
            method.ReturnType = new CodeTypeReference("PixelInputDeclaration");
            method.Parameters.Add(new CodeParameterDeclarationExpression(new CodeTypeReference("VertexConstantDeclaration"), "constants"));
            method.Parameters.Add(new CodeParameterDeclarationExpression(new CodeTypeReference("VertexInputDeclaration"), "input"));
            method.Attributes = MemberAttributes.Public | MemberAttributes.Static;
            method.Statements.Add(new CodeVariableDeclarationStatement(new CodeTypeReference("PixelInputDeclaration"), "output"));
            method.Statements.Add(new CodeAssignStatement(new CodeVariableReferenceExpression("output"), new CodeObjectCreateExpression(new CodeTypeReference("PixelInputDeclaration"))));
            method.Statements.Add(new CodeMethodReturnStatement(new CodeVariableReferenceExpression("output")));
            return method;
        }
        static CodeMemberMethod CreatePixelShader()
        {
            var method = new CodeMemberMethod();
            method.Name = "PixelShader";
            method.ReturnType = new CodeTypeReference(typeof(Vector4F));
            method.Parameters.Add(new CodeParameterDeclarationExpression(new CodeTypeReference("PixelConstantDeclaration"), "constants"));
            method.Parameters.Add(new CodeParameterDeclarationExpression(new CodeTypeReference("PixelInputDeclaration"), "input"));
            method.Attributes = MemberAttributes.Public | MemberAttributes.Static;
            method.Statements.Add(new CodeMethodReturnStatement(new CodeObjectCreateExpression(typeof(Vector4F), new CodeExpression[] {
                new CodePrimitiveExpression(1.0f),
                new CodePrimitiveExpression(0.0f),
                new CodePrimitiveExpression(0.0f),
                new CodePrimitiveExpression(1.0f),
                })));
            return method;
        }
        static CodeTypeDeclaration CreatePipeline()
        {
            var maketype = new CodeTypeDeclaration("Pipeline");
            maketype.Attributes = MemberAttributes.Static;
            maketype.IsClass = true;
            maketype.Members.Add(CreateVertexShader());
            maketype.Members.Add(CreatePixelShader());
            return maketype;
        }
        static CodeNamespace CreateRendererNamespace()
        {
            var makens = new CodeNamespace("RenderToy.Generated");
            makens.Types.Add(CreateVertexConstantDeclaration());
            makens.Types.Add(CreateVertexInputDeclaration());
            makens.Types.Add(CreatePixelConstantDeclaration());
            makens.Types.Add(CreatePixelInputDeclaration());
            makens.Types.Add(CreatePipeline());
            return makens;
        }
        static CodeCompileUnit CreatedRendererCompileUnit()
        {
            var makecu = new CodeCompileUnit();
            makecu.Namespaces.Add(CreateRendererNamespace());
            return makecu;
        }
        public static CodeCompileUnit GenerateRenderer()
        {
            return CreatedRendererCompileUnit();
        }
        public static string DoString(CodeCompileUnit codecompileunit)
        {
            var provider = new CSharpCodeProvider();
            var writer = new StringWriter();
            provider.GenerateCodeFromCompileUnit(codecompileunit, writer, new CodeGeneratorOptions { BlankLinesBetweenMembers = false });
            return writer.ToString();
        }
        public static CompilerResults DoCompile(CodeCompileUnit codecompileunit)
        {
            var provider = new CSharpCodeProvider();
            var compilerparameters = new CompilerParameters();
            compilerparameters.GenerateInMemory = true;
            compilerparameters.ReferencedAssemblies.Add("RenderToyCoreLibrary.dll");
            var result = provider.CompileAssemblyFromDom(compilerparameters, codecompileunit);
            if (result.Errors.HasErrors)
            {
                throw new Exception(result.Errors[0].ErrorText);
            }
            return result;
        }
    }
}