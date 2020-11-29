#pragma once

#include "IDevice3D_D3D12.h"
#include "IVertexBuffer.h"
#include <atlbase.h>
#include <d3d12.h>
#include <vector>

namespace Arcturus
{
    class IShader_D3D12 : public IShader
    {
    public:
        IShader_D3D12(IDevice3D_D3D12* owner);
        IDevice3D_D3D12* m_owner;
        std::vector<uint8_t> m_vertexShader;
        std::vector<uint8_t> m_pixelShader;
        CComPtr<ID3D12PipelineState> m_pipelineState;
    };
}