#pragma once

#include "IBuffer_D3D12.h"
#include "IConstantBuffer.h"

namespace Arcturus
{
    class IConstantBuffer_D3D12 : public IBuffer_D3D12, public IConstantBuffer
    {
    public:
        IConstantBuffer_D3D12(IDevice3D_D3D12* owner, uint32_t dataSize, const void* data);
    };
}