#pragma once

#include "IBuffer_D3D12.h"
#include "IIndexBuffer.h"

namespace Arcturus
{
    class IIndexBuffer_D3D12 : public IBuffer_D3D12, public IIndexBuffer
    {
    public:
        IIndexBuffer_D3D12(IDevice3D_D3D12* owner, uint32_t dataSize, const void* data);
        D3D12_INDEX_BUFFER_VIEW m_descIndexBufferView;
    };
}