////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace RenderToy
{
    public class HLSLGenerator
    {
        public static string Emit(Expression expr)
        {
            return new HLSLGenerator().EmitMain(expr);
        }
        string EmitMain(Expression expr)
        {
            var writer = new StringWriter();
            var codewriter = new CodeWriter(writer);
            // Emit the interpolator definition.
            writer.WriteLine(
@"
struct VS_INPUT
{
    float4 Position : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 Color : COLOR;
    float3 Tangent : TANGENT;
    float3 Bitangent : BINORMAL;
};");
            writer.WriteLine();
            // Find all the lambdas.
            var flatten = new ExpressionCounter();
                flatten.Visit(expr);
                LambdaMap =
                    flatten.Found.Keys
                    .Select(i => i)
                    .OfType<LambdaExpression>()
                    .Select((e, i) => new { Lambda = e, Name = e.Name == null ? "_Fn" + i : e.Name })
                    .ToDictionary(k => k.Lambda, v => v.Name);
            // Emit prototypes for all lambdas.
            foreach (var emitproto in LambdaMap)
            {
                codewriter.Write(EmitType(emitproto.Key.ReturnType) + " " + emitproto.Value);
                codewriter.Write("(");
                bool more = false;
                foreach (var arg in emitproto.Key.Parameters)
                {
                    if (more) codewriter.Write(", ");
                    more = true;
                    codewriter.Write(EmitType(arg.Type));
                    codewriter.Write(" ");
                    codewriter.Write(arg.Name);
                }
                codewriter.Write(");");
            }
            codewriter.WriteLine();
            // Emit bodies for all the lambdas.
            foreach (var emitproto in LambdaMap)
            {
                codewriter.Write(EmitType(emitproto.Key.ReturnType) + " " + emitproto.Value);
                codewriter.Write("(");
                bool more = false;
                foreach (var arg in emitproto.Key.Parameters)
                {
                    if (more) codewriter.Write(", ");
                    more = true;
                    codewriter.Write(EmitType(arg.Type));
                    codewriter.Write(" ");
                    codewriter.Write(arg.Name);
                }
                codewriter.Write(")");
                codewriter.Write("{");
                EmitMain(emitproto.Key.Body, codewriter);
                codewriter.Write("}");
            }
            // Emit the body code of the shader.
            codewriter.Write("float4 ps(VS_INPUT interpolators) : SV_Target");
            codewriter.Write("{");
            EmitMain(expr, codewriter);
            codewriter.Write("}");
            return writer.ToString();
        }
        void EmitMain(Expression expr, CodeWriter codewriter)
        {
            if (expr is BlockExpression block)
            {
                Emit(block, codewriter);
            }
            else
            {
                Emit(Expression.Block(expr), codewriter);
            }
        }
        void Emit(Expression expr, CodeWriter codewriter)
        {
            switch (expr.NodeType)
            {
                case ExpressionType.Add:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" + ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.And:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" & ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.Assign:
                    {
                        var cast = (BinaryExpression)expr;
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" = ");
                        Emit(cast.Right, codewriter);
                    }
                    break;
                case ExpressionType.Block:
                    {
                        var cast = (BlockExpression)expr;
                        foreach (var param in cast.Variables)
                        {
                            codewriter.Write(EmitType(param.Type) + " " + param.Name + ";");
                        }
                        var listexpr = cast.Expressions.ToList();
                        foreach (var assign in listexpr)
                        {
                            if (assign == listexpr.Last()) codewriter.Write("return ");
                            Emit(assign, codewriter);
                            codewriter.Write(";");
                        }
                    }
                    break;
                case ExpressionType.Call:
                    {
                        var cast = (MethodCallExpression)expr;
                        codewriter.Write(EmitMethod(cast.Method));
                        codewriter.Write("(");
                        bool more = false;
                        foreach (var arg in cast.Arguments)
                        {
                            if (more) codewriter.Write(",");
                            more = true;
                            Emit(arg, codewriter);
                        }
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.Conditional:
                    {
                        var cast = (ConditionalExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Test, codewriter);
                        codewriter.Write(" ? ");
                        Emit(cast.IfTrue, codewriter);
                        codewriter.Write(" : ");
                        Emit(cast.IfFalse, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.Constant:
                    {
                        var cast = (ConstantExpression)expr;
                        codewriter.Write(cast.Value.ToString());
                    }
                    break;
                case ExpressionType.Convert:
                    {
                        var cast = (UnaryExpression)expr;
                        codewriter.Write("((");
                        codewriter.Write(EmitType(cast.Type));
                        codewriter.Write(")");
                        Emit(cast.Operand, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.Divide:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" / ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.Equal:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" == ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.ExclusiveOr:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" ^ ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.Invoke:
                    {
                        var cast = (InvocationExpression)expr;
                        codewriter.Write(LambdaMap[(LambdaExpression)cast.Expression]);
                        codewriter.Write("(");
                        bool more = false;
                        foreach (var arg in cast.Arguments)
                        {
                            if (more) codewriter.Write(", ");
                            more = true;
                            Emit(arg, codewriter);
                        }
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.LeftShift:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" << ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.LessThan:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" < ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.MemberAccess:
                    {
                        var cast = (MemberExpression)expr;
                        codewriter.Write(EmitField(cast.Member));
                    }
                    break;
                case ExpressionType.Multiply:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" * ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.New:
                    {
                        var cast = (NewExpression)expr;
                        codewriter.Write(EmitType(cast.Type));
                        codewriter.Write("(");
                        bool more = false;
                        foreach (var arg in cast.Arguments)
                        {
                            if (more) codewriter.Write(", ");
                            more = true;
                            Emit(arg, codewriter);
                        }
                        codewriter.Write(")");
                    }
                    break;
                case ExpressionType.Parameter:
                    {
                        var cast = (ParameterExpression)expr;
                        codewriter.Write(cast.Name);
                    }
                    break;
                case ExpressionType.Subtract:
                    {
                        var cast = (BinaryExpression)expr;
                        codewriter.Write("(");
                        Emit(cast.Left, codewriter);
                        codewriter.Write(" - ");
                        Emit(cast.Right, codewriter);
                        codewriter.Write(")");
                    }
                    break;
                default: throw new NotSupportedException("Don't know how to generate HLSL for '" + expr.NodeType + "'.");
            }
        }
        static string EmitField(MemberInfo member)
        {
            if (member.DeclaringType == typeof(EvalContext) && member.Name == "U") return "interpolators.TexCoord.x";
            if (member.DeclaringType == typeof(EvalContext) && member.Name == "V") return "interpolators.TexCoord.y";
            else throw new NotSupportedException("Don't know how to generate HLSL for member '" + member + "'.");
        }
        static string EmitMethod(MethodInfo method)
        {
            if (method.DeclaringType == typeof(Math) && method.Name == "Floor") return "floor";
            if (method.DeclaringType == typeof(Math) && method.Name == "Pow") return "pow";
            if (method.DeclaringType == typeof(Math) && method.Name == "Sin") return "sin";
            else throw new NotSupportedException("Don't know how to generate HLSL for method '" + method + "'.");
        }
        static string EmitType(Type type)
        {
            if (type == typeof(System.Double)) return "float";
            else if (type == typeof(System.Int32)) return "int";
            else if (type == typeof(Vector4D)) return "float4";
            else throw new NotSupportedException("Don't know how to generate HLSL for type '" + type + "'.");
        }
        IDictionary<LambdaExpression, string> LambdaMap;
        class CodeWriter
        {
            public CodeWriter(StringWriter writer)
            {
                Writer = writer;
            }
            public void Write(char c)
            {
                if (c == '{')
                {
                    WriteLine();
                    Writer.Write(c);
                    ++indent;
                    WriteLine();
                }
                else if (c == '}')
                {
                    indent = Math.Max(0, indent - 1);
                    WriteLine();
                    Writer.Write(c);
                    WriteLine();
                }
                else if (c == ';')
                {
                    Writer.Write(c);
                    WriteLine();
                }
                else
                {
                    Writer.Write(c);
                }
            }
            public void Write(string s)
            {
                if (s == null)
                {
                    Write("NULL");
                    return;
                }
                foreach (var c in s)
                {
                    Write(c);
                }
            }
            public void WriteLine()
            {
                Writer.WriteLine();
                for (int i = 0; i < indent; ++i)
                {
                    Write('\t');
                }
            }
            StringWriter Writer;
            int indent = 0;
        }
    }
}