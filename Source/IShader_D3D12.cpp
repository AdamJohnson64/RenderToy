#include "ErrorD3D.h"
#include "IShader_D3D12.h"
#include "Vector.h"

#include <d3dcompiler.h>

namespace
{

    const char SHADER_CODE[] =
        R"TEXT(
cbuffer CONSTANTS : register(b0)
{
    float4x4 vertexTransform;
};

struct VS_INPUT
{
    float4 Position : POSITION;
    float4 Color    : COLOR0;
};

struct PS_INPUT
{
    float4 Position : SV_Position;
    float4 Color : COLOR0;
};

PS_INPUT vs(VS_INPUT v)
{
    PS_INPUT result;
    result.Position = mul(vertexTransform, v.Position);
    result.Color = v.Color;
    return result;
}

float4 ps(PS_INPUT p) : SV_Target0
{
    return p.Color;
}
)TEXT";

}

namespace Arcturus
{
    IShader_D3D12::IShader_D3D12(IDevice3D_D3D12* owner) : m_owner(owner)
    {
        // Compile the vertex shader.
        {
            AutoRelease<ID3DBlob> blobCode, blobError;
            if (FAILED(D3DCompile(SHADER_CODE, sizeof(SHADER_CODE), nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0, &blobCode, &blobError)))
            {
                const char* errorString = static_cast<const char*>(blobError->GetBufferPointer());
                throw std::exception("Failed to compile vertex shader.");
            }
            m_vertexShader.resize(blobCode->GetBufferSize());
            memcpy(&m_vertexShader[0], blobCode->GetBufferPointer(), blobCode->GetBufferSize());
        }
        // Compile the pixel shader.
        {
            AutoRelease<ID3DBlob> blobCode, blobError;
            if (FAILED(D3DCompile(SHADER_CODE, sizeof(SHADER_CODE), nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0, &blobCode, &blobError)))
            {
                const char* errorString = static_cast<const char*>(blobError->GetBufferPointer());
                throw std::exception("Failed to compile pixel shader.");
            }
            m_pixelShader.resize(blobCode->GetBufferSize());
            memcpy(&m_pixelShader[0], blobCode->GetBufferPointer(), blobCode->GetBufferSize());
        }
        // Create a complete graphics pipeline.
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC descPipeline = {};
            descPipeline.pRootSignature = m_owner->m_rootSignature;
            descPipeline.VS.pShaderBytecode = &m_vertexShader[0];
            descPipeline.VS.BytecodeLength = m_vertexShader.size();
            descPipeline.PS.pShaderBytecode = &m_pixelShader[0];
            descPipeline.PS.BytecodeLength = m_pixelShader.size();
            descPipeline.BlendState.RenderTarget[0].BlendEnable = TRUE;
            descPipeline.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
            descPipeline.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
            descPipeline.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
            descPipeline.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_SRC_ALPHA;
            descPipeline.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
            descPipeline.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
            descPipeline.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
            descPipeline.SampleMask = 0xFFFFFFFF;
            descPipeline.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            descPipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            D3D12_INPUT_ELEMENT_DESC descElement[2] = {};
            descElement[0].SemanticName = "POSITION";
            descElement[0].Format = DXGI_FORMAT_R32G32_FLOAT;
            descElement[0].AlignedByteOffset = offsetof(Vertex, Position);
            descElement[1].SemanticName = "COLOR";
            descElement[1].Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            descElement[1].AlignedByteOffset = offsetof(Vertex, Color);
            descPipeline.InputLayout.NumElements = _countof(descElement);
            descPipeline.InputLayout.pInputElementDescs = descElement;
            descPipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            descPipeline.NumRenderTargets = 1;
            descPipeline.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
            descPipeline.SampleDesc.Count = 1;
            TRYD3D(m_owner->m_device->CreateGraphicsPipelineState(&descPipeline, __uuidof(ID3D12PipelineState), (void**)&m_pipelineState));
        }
    }
}