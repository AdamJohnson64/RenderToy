#pragma once

#include "IBuffer_Vulkan.h"
#include "IIndexBuffer.h"

namespace Arcturus
{
    class IIndexBuffer_Vulkan : public IBuffer_Vulkan, public IIndexBuffer
    {
    public:
        IIndexBuffer_Vulkan(IDevice3D_Vulkan* owner, uint32_t dataSize, const void* data);
    };
}