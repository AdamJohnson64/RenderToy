#include "IConstantBuffer_Vulkan.h"

namespace Arcturus
{
    IConstantBuffer_Vulkan::IConstantBuffer_Vulkan(IDevice3D_Vulkan* owner, uint32_t dataSize, const void* data) : IBuffer_Vulkan(owner, dataSize, data, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
    {
    }
}