#pragma once

#include "VulkanInclude.h"

#include <stdint.h>

namespace Arcturus
{
    class IDevice3D_Vulkan;

    class IBuffer_Vulkan
    {
    public:
        IBuffer_Vulkan(IDevice3D_Vulkan* owner, uint32_t dataSize, const void* data, VkBufferUsageFlags usage);
        ~IBuffer_Vulkan();
        IDevice3D_Vulkan* m_owner;
        uint32_t m_byteSize;
        VkBuffer m_vkBuffer;
        VkDeviceMemory m_vkDeviceMemory;
    };
}