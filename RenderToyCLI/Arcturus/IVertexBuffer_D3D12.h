#pragma once

#include "IBuffer_D3D12.h"
#include "IVertexBuffer.h"

namespace Arcturus
{
    class IVertexBuffer_D3D12 : public IBuffer_D3D12, public IVertexBuffer
    {
    public:
        IVertexBuffer_D3D12(IDevice3D_D3D12* owner, uint32_t dataSize, uint32_t strideSize, const void* data);
        uint32_t m_strideSize;
        D3D12_VERTEX_BUFFER_VIEW m_descVertexBufferView;
    };
}