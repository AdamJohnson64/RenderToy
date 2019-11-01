#pragma once

#include "ITexture.h"

#include <stdint.h>

namespace Arcturus
{
    class IDevice3D_Vulkan;

    class ITexture_Vulkan : public ITexture
    {
    public:
        ITexture_Vulkan(IDevice3D_Vulkan* owner, uint32_t width, uint32_t height, const void* data);
        ~ITexture_Vulkan();
        IDevice3D_Vulkan*   m_owner;
        VkImage             m_vkImage;
        VkDeviceMemory      m_vkImageMemory;
        VkImageView         m_vkImageView;
    };
}