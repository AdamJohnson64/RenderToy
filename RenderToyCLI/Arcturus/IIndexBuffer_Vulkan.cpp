#include "IIndexBuffer_Vulkan.h"

namespace Arcturus
{
    IIndexBuffer_Vulkan::IIndexBuffer_Vulkan(IDevice3D_Vulkan* owner, uint32_t dataSize, const void* data) : IBuffer_Vulkan(owner, dataSize, data, VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
    {
    }
}