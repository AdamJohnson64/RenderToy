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
    result.Normal = input.Normal;
    result.Color = input.Color;
    result.TexCoord = input.TexCoord;
    result.Tangent = input.Tangent;
    result.Bitangent = input.Bitangent;
    result.EyeVector = float3(TransformCamera[0].w, TransformCamera[1].w, TransformCamera[2].w) - input.Position.xyz;
    return result;
}

";
        public static readonly string D3DPixelShaderCode =
@"// Direct3D Standard Pixel Shader
float4 ps(VS_OUTPUT input) : SV_Target {
    return float4(1, 1, 1, 1);
}

";
        #endregion
        #region - Section : Direct3D9 -
        public static readonly string D3D9Constants =
@"// Direct3D9 Standard Constants
float4x4 TransformCamera : register(c0);
float4x4 TransformModel : register(c4);
float4x4 TransformView : register(c8);
float4x4 TransformProjection : register(c12);
float4x4 TransformModelViewProjection : register(c16);
sampler2D SamplerAlbedo : register(s0);
sampler2D SamplerMask : register(s1);
sampler2D SamplerBump : register(s2);
sampler2D SamplerDisplacement : register(s3);

";
        public static readonly string D3D9PixelShaderCode =
@"// Direct3D9 Standard Pixel Shader
float4 ps(VS_OUTPUT input) : SV_Target {
    ////////////////////////////////////////////////////////////////////////////////
    // Stencil Mask
    if (tex2D(SamplerMask, input.TexCoord).r < 0.5) discard;

    ////////////////////////////////////////////////////////////////////////////////
    // Reconstruct Tangent Basis
    float3x3 tbn = {input.Tangent, input.Bitangent, input.Normal};

    ////////////////////////////////////////////////////////////////////////////////
    // Displacement Mapping (Steep Parallax)
    float height = 1.0;
    float bumpScale = 0.02;
    float numSteps = 20;
    float2 offsetCoord = input.TexCoord.xy;
    float sampledHeight = tex2D(SamplerDisplacement, offsetCoord).r;
    float3 tangentSpaceEye = mul(input.EyeVector, transpose(tbn));
    numSteps = lerp(numSteps * 2, numSteps, normalize(tangentSpaceEye).z);
    float step = 1.0 / numSteps;
    float2 delta = -float2(tangentSpaceEye.x, tangentSpaceEye.y) * bumpScale / (tangentSpaceEye.z * numSteps);
    int maxiter = 50;
    int iter = 0;
    while (sampledHeight < height && iter < maxiter) {
        height -= step;
        offsetCoord += delta;
        sampledHeight = tex2D(SamplerDisplacement, offsetCoord).r;
        ++iter;
    }
    height = sampledHeight;

    ////////////////////////////////////////////////////////////////////////////////
    // Bump Mapping Normal
    float3 bump = normalize(tex2D(SamplerBump, offsetCoord).rgb * 2 - 1);
    float3 normal = mul(bump, tbn);

    ////////////////////////////////////////////////////////////////////////////////
    // Simple Lighting
    float light = clamp(dot(normal, normalize(float3(1,1,1))), 0, 1);

    ////////////////////////////////////////////////////////////////////////////////
    // Final Color
    return float4(light * tex2D(SamplerAlbedo, offsetCoord).rgb, 1);
}

";
        public static readonly string DX9Full =
            D3D9Constants +
            D3DVertexInputStruct +
            D3DVertexOutputStruct +
            D3DVertexShaderCode +
            D3D9PixelShaderCode;
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

";
        public readonly static string DX11Simple =
            D3D11Constants +
            D3DVertexInputStruct +
            D3DVertexOutputStruct +
            D3DVertexShaderCode +
            D3DPixelShaderCode;
        #endregion
        #region - Section : Direct3D12 -
        public readonly static string D3D12Constants =
@"#define CommonRoot \
""RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT),"" \
""RootConstants(num32BitConstants=16, b0, space=0, visibility=SHADER_VISIBILITY_ALL)""

cbuffer Constants : register(b0)
{
    float4x4 TransformModelViewProjection;
    float4x4 TransformCamera;
    float4x4 TransformModel;
    float4x4 TransformView;
    float4x4 TransformProjection;
};

";
        public readonly static string DX12Simple =
            D3D12Constants +
            D3DVertexInputStruct +
            D3DVertexOutputStruct +
@"[RootSignature(CommonRoot)]
" +
            D3DVertexShaderCode +
            D3DPixelShaderCode;
        #endregion
    }
}