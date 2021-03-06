﻿using RenderToy.Expressions;

namespace RenderToy.Shaders
{
    public static class HLSL
    {
        #region - Section : Direct3D Common -
        public static readonly string D3DVertexInputStruct =
@"// Direct3D Standard Vertex Shader Input
struct VS_INPUT {
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR;
    float2 TexCoord : TEXCOORD0;
    float3 Tangent : TANGENT;
    float3 Bitangent : BINORMAL;
};

";
        public static readonly string D3DVertexOutputStruct =
@"// Direct3D Standard Vertex Shader Output
struct VS_OUTPUT {
    float4 Position : SV_Position;
    float3 Normal : NORMAL;
    float4 Color : COLOR;
    float2 TexCoord : TEXCOORD0;
    float3 Tangent : TANGENT;
    float3 Bitangent : BINORMAL;
    float3 EyeVector : TEXCOORD1;
};

";
        public static readonly string D3DVertexShaderCode =
@"// Direct3D Standard Vertex Shader
VS_OUTPUT vs(VS_INPUT input) {
    VS_OUTPUT result;
    result.Position = mul(TransformModelViewProjection, float4(input.Position, 1));
    result.Normal = normalize(mul(TransformModel, float4(input.Normal, 0)).xyz);
    result.Color = input.Color;
    result.TexCoord = input.TexCoord;
    result.Tangent = normalize(mul(TransformModel, float4(input.Tangent, 0)).xyz);
    result.Bitangent = normalize(mul(TransformModel, float4(input.Bitangent, 0)).xyz);
    result.EyeVector = float3(TransformCamera[0].w, TransformCamera[1].w, TransformCamera[2].w) - mul(TransformModel, input.Position.xyz);
    return result;
}

";
        public static readonly string D3DPixelShaderCode =
@"// Direct3D Standard Pixel Shader
float4 ps(VS_OUTPUT input) : SV_Target {
    ////////////////////////////////////////////////////////////////////////////////
    // Stencil Mask
    if (SampleTexture2D(TextureMask, input.TexCoord).r < 0.5) discard;

    ////////////////////////////////////////////////////////////////////////////////
    // Reconstruct Tangent Basis
    float3x3 tbn = {input.Tangent, input.Bitangent, input.Normal};

    ////////////////////////////////////////////////////////////////////////////////
    // Displacement Mapping (Steep Parallax)
    float height = 1.0;
    float bumpScale = 0.005;
    float numSteps = 20;
    float2 offsetCoord = input.TexCoord.xy;
    float sampledHeight = SampleTexture2D(TextureDisplacement, offsetCoord).r;
    float3 tangentSpaceEye = mul(input.EyeVector, transpose(tbn));
    numSteps = lerp(numSteps * 2, numSteps, normalize(tangentSpaceEye).z);
    float step = 1.0 / numSteps;
    float2 delta = -float2(tangentSpaceEye.x, tangentSpaceEye.y) * bumpScale / (tangentSpaceEye.z * numSteps);
    int maxiter = 50;
    int iter = 0;
    while (sampledHeight < height && iter < maxiter) {
        height -= step;
        offsetCoord += delta;
        sampledHeight = SampleTexture2D(TextureDisplacement, offsetCoord).r;
        ++iter;
    }
    height = sampledHeight;

    ////////////////////////////////////////////////////////////////////////////////
    // Bump Mapping Normal
    float3 bump = normalize(SampleTexture2D(TextureBump, offsetCoord).rgb * 2 - 1);
    float3 normal = mul(bump, tbn);

    ////////////////////////////////////////////////////////////////////////////////
    // Simple Lighting
    float light = clamp(dot(normal, normalize(float3(1,1,-1))), 0, 1);

    ////////////////////////////////////////////////////////////////////////////////
    // Final Color
    return float4(light * SampleTexture2D(TextureAlbedo, offsetCoord).rgb, 1);
}

";
        public static readonly string D3DPixelShaderCodeEnvironment =
@"// Direct3D Simple Pixel Shader
float4 ps(VS_OUTPUT input) : SV_Target {
    return float4(SampleTextureCUBE(TextureEnvironment, input.Normal).rgb, 1);
}

";
        public static readonly string D3DPixelShaderCodeSimple =
@"// Direct3D Simple Pixel Shader
float4 ps(VS_OUTPUT input) : SV_Target {
    return float4(1, 1, 1, 1);
}

";
        public static readonly string D3DPixelShaderCodeUnlit =
@"// Direct3D Simple Pixel Shader
float4 ps(VS_OUTPUT input) : SV_Target {
    // 1 TAP - This is going to look mighty ugly in VR.
    //return float4(SampleTexture2D(TextureAlbedo, input.TexCoord.xy).rgb, 1);
    // X*Y TAP - Crank this up as high as you like for in-shader anti-aliasing of a single MIP texture.
    const int samplesU = 9;
    const int samplesV = 9;
    const float2 duvdx = ddx(input.TexCoord.xy); 
    const float2 duvdy = ddy(input.TexCoord.xy);
    float4 acc = float4(0, 0, 0, 0);
    for (int y = 0; y < samplesV; ++y) {
        for (int x = 0; x < samplesU; ++x) {
            acc += SampleTexture2D(TextureAlbedo, input.TexCoord.xy + duvdx * lerp(-0.5, 0.5, (x + 0.5) / samplesU) + duvdy * lerp(-0.5, 0.5, (y + 0.5) / samplesV));
        }
    }
    return acc / (samplesU * samplesV);
}

";
        #endregion
        #region - Section : Direct3D11 -
        public readonly static string D3D11Constants =
@"// Direct3D11 Standard Constants
cbuffer Constants : register(b0)
{
    float4x4 TransformModelViewProjection;
    float4x4 TransformCamera;
    float4x4 TransformModel;
    float4x4 TransformView;
    float4x4 TransformProjection;
};

SamplerState Sampler : register(s0);
SamplerState Sampler2 : register(s1);

Texture2D TextureAlbedo : register(t0);
Texture2D TextureMask : register(t1);
Texture2D TextureBump : register(t2);
Texture2D TextureDisplacement : register(t3);
TextureCube TextureEnvironment : register(t4);

float4 SampleTexture2D(texture2D s, float2 uv) { return s.Sample(Sampler, uv); }
float4 SampleTextureCUBE(textureCUBE s, float3 uvw) { return s.Sample(Sampler, uvw); }

";
        public readonly static string D3D11Standard =
            D3D11Constants +
            D3DVertexInputStruct +
            D3DVertexOutputStruct +
            D3DVertexShaderCode +
            D3DPixelShaderCode;
        public readonly static string D3D11Environment =
            D3D11Constants +
            D3DVertexInputStruct +
            D3DVertexOutputStruct +
            D3DVertexShaderCode +
            D3DPixelShaderCodeEnvironment;
        public readonly static string D3D11Unlit =
            D3D11Constants +
            D3DVertexInputStruct +
            D3DVertexOutputStruct +
            D3DVertexShaderCode +
            D3DPixelShaderCodeUnlit;
        #endregion
        #region - Section : Compiled Results -
        public static readonly byte[] D3D11VS = HLSLExtensions.CompileHLSL(HLSL.D3D11Standard, "vs", "vs_5_0");
        public static readonly byte[] D3D11PS = HLSLExtensions.CompileHLSL(HLSL.D3D11Standard, "ps", "ps_5_0");
        public static readonly byte[] D3D11PSEnvironment = HLSLExtensions.CompileHLSL(HLSL.D3D11Environment, "ps", "ps_5_0");
        public static readonly byte[] D3D11PSUnlit = HLSLExtensions.CompileHLSL(HLSL.D3D11Unlit, "ps", "ps_5_0");
        #endregion
    }
}