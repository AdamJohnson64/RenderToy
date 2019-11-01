#include "ErrorVK.h"
#include "IDevice3D_Vulkan.h"
#include "IRenderTarget_Vulkan.h"

namespace Arcturus
{
    IRenderTarget_Vulkan::IRenderTarget_Vulkan(IDevice3D_Vulkan* owner, const RenderTargetDeclaration& declaration, HANDLE d3d11) : m_owner(owner), m_width(declaration.width), m_height(declaration.height)
    {
        {
            VkImageCreateInfo descImageCreateInfo = {};
            descImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            descImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            descImageCreateInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
            descImageCreateInfo.extent.width = declaration.width;
            descImageCreateInfo.extent.height = declaration.height;
            descImageCreateInfo.extent.depth = 1;
            descImageCreateInfo.mipLevels = 1;
            descImageCreateInfo.arrayLayers = 1;
            descImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            descImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            descImageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            descImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            descImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            TRYVK(vkCreateImage(m_owner->m_vkDevice, &descImageCreateInfo, nullptr, &m_vkImage));
        }
        {
            // Determine the GPU memory requirements for this image.
            VkMemoryRequirements descMemoryRequirements = {};
            vkGetImageMemoryRequirements(m_owner->m_vkDevice, m_vkImage, &descMemoryRequirements);
            // Allocate the memory for the image via sharing.
            VkImportMemoryWin32HandleInfoKHR descImportMemoryWin32HandleInfoKHR = {};
            descImportMemoryWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
            descImportMemoryWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
            descImportMemoryWin32HandleInfoKHR.handle = d3d11;
            VkMemoryAllocateInfo descMemoryAllocateInfo = {};
            descMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            descMemoryAllocateInfo.pNext = &descImportMemoryWin32HandleInfoKHR;
            descMemoryAllocateInfo.allocationSize = descMemoryRequirements.size;
            descMemoryAllocateInfo.memoryTypeIndex = m_owner->m_memoryTypeDevice;
            TRYVK(m_owner->m_dynamicLoader.vkAllocateMemory(m_owner->m_vkDevice, reinterpret_cast<VkMemoryAllocateInfo*>(&descMemoryAllocateInfo), nullptr, &m_vkDeviceMemory));
        }
        // Bind our D3D12 texture resource into this image.
        TRYVK(vkBindImageMemory(m_owner->m_vkDevice, m_vkImage, m_vkDeviceMemory, 0));
        {
            VkImageViewCreateInfo descImageViewCreateInfo = {};
            descImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            descImageViewCreateInfo.image = m_vkImage;
            descImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            descImageViewCreateInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
            descImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
            descImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
            descImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
            descImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
            descImageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            descImageViewCreateInfo.subresourceRange.levelCount = 1;
            descImageViewCreateInfo.subresourceRange.layerCount = 1;
            TRYVK(vkCreateImageView(m_owner->m_vkDevice, &descImageViewCreateInfo, nullptr, &m_vkImageView));
        }
        // Create the framebuffer (depends on vkImageView).
        {
            VkFramebufferCreateInfo descFramebufferCreateInfo = {};
            descFramebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            descFramebufferCreateInfo.renderPass = m_owner->m_vkRenderPass;
            descFramebufferCreateInfo.attachmentCount = 1;
            descFramebufferCreateInfo.pAttachments = &m_vkImageView;
            descFramebufferCreateInfo.width = m_width;
            descFramebufferCreateInfo.height = m_height;
            descFramebufferCreateInfo.layers = 1;
            TRYVK(vkCreateFramebuffer(m_owner->m_vkDevice, &descFramebufferCreateInfo, nullptr, &m_vkFramebuffer));
        }
    }

    IRenderTarget_Vulkan::~IRenderTarget_Vulkan()
    {
        vkDestroyFramebuffer(m_owner->m_vkDevice, m_vkFramebuffer, nullptr);
        m_vkFramebuffer = VK_NULL_HANDLE;
        vkDestroyImageView(m_owner->m_vkDevice, m_vkImageView, nullptr);
        m_vkImageView = VK_NULL_HANDLE;
        vkDestroyImage(m_owner->m_vkDevice, m_vkImage, nullptr);
        m_vkImage = VK_NULL_HANDLE;
        vkFreeMemory(m_owner->m_vkDevice, m_vkDeviceMemory, nullptr);
        m_vkDeviceMemory = VK_NULL_HANDLE;
    }
}