#pragma once

#include "IBuffer_Vulkan.h"
#include "IConstantBuffer.h"

namespace Arcturus
{
    class IConstantBuffer_Vulkan : public IBuffer_Vulkan, public IConstantBuffer
    {
    public:
        IConstantBuffer_Vulkan(IDevice3D_Vulkan* owner, uint32_t dataSize, const void* data);
    };
}