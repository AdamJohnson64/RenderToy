#pragma once

#include "IBuffer_Vulkan.h"
#include "IVertexBuffer.h"

namespace Arcturus
{
    class IVertexBuffer_Vulkan : public IBuffer_Vulkan, public IVertexBuffer
    {
    public:
        IVertexBuffer_Vulkan(IDevice3D_Vulkan* owner, uint32_t dataSize, const void* data);
    };
}