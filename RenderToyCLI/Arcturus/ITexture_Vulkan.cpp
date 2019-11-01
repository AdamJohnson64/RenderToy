#include "ErrorVK.h"
#include "IDevice3D_Vulkan.h"
#include "ITexture_Vulkan.h"

#include "VulkanInclude.h"

namespace Arcturus
{
    ITexture_Vulkan::ITexture_Vulkan(IDevice3D_Vulkan* owner, uint32_t width, uint32_t height, const void* data) : m_owner(owner)
    {
        {
            {
                VkImageCreateInfo descImageCreateInfo = {};
                descImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                descImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
                descImageCreateInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
                descImageCreateInfo.extent.width = width;
                descImageCreateInfo.extent.height = height;
                descImageCreateInfo.extent.depth = 1;
                descImageCreateInfo.mipLevels = 1;
                descImageCreateInfo.arrayLayers = 1;
                descImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
                descImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
                descImageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
                descImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                descImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                TRYVK(vkCreateImage(m_owner->m_vkDevice, &descImageCreateInfo, nullptr, &m_vkImage));
            }
            {
                VkMemoryRequirements descMemoryRequirements;
                vkGetImageMemoryRequirements(m_owner->m_vkDevice, m_vkImage, &descMemoryRequirements);
                {
                    VkMemoryAllocateInfo descMemoryAllocateInfo = {};
                    descMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    descMemoryAllocateInfo.allocationSize = descMemoryRequirements.size;
                    descMemoryAllocateInfo.memoryTypeIndex = m_owner->m_memoryTypeDevice;
                    TRYVK(vkAllocateMemory(m_owner->m_vkDevice, &descMemoryAllocateInfo, nullptr, &m_vkImageMemory));
                }
            }
            TRYVK(vkBindImageMemory(m_owner->m_vkDevice, m_vkImage, m_vkImageMemory, 0));
        }
        {
            VkBuffer vkStagingBuffer = VK_NULL_HANDLE;
            VkDeviceMemory vkStagingBufferMemory = VK_NULL_HANDLE;
            {
                VkBufferCreateInfo descBufferCreateInfo = {};
                descBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                descBufferCreateInfo.size = sizeof(uint32_t) * width * height;
                descBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                descBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                TRYVK(vkCreateBuffer(m_owner->m_vkDevice, &descBufferCreateInfo, nullptr, &vkStagingBuffer));
            }
            {
                VkMemoryAllocateInfo descMemoryAllocateInfo = {};
                descMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                descMemoryAllocateInfo.allocationSize = sizeof(uint32_t) * width * height;
                descMemoryAllocateInfo.memoryTypeIndex = m_owner->m_memoryTypeShared;
                TRYVK(vkAllocateMemory(m_owner->m_vkDevice, &descMemoryAllocateInfo, nullptr, &vkStagingBufferMemory));
            }
            TRYVK(vkBindBufferMemory(m_owner->m_vkDevice, vkStagingBuffer, vkStagingBufferMemory, 0));
            {
                void* pData;
                TRYVK(vkMapMemory(owner->m_vkDevice, vkStagingBufferMemory, 0, sizeof(uint32_t) * width * height, 0, &pData));
                memcpy(pData, data, sizeof(uint32_t) * width * height);
                vkUnmapMemory(owner->m_vkDevice, vkStagingBufferMemory);
            }
            // Copy the buffer over to the texture.
            VkCommandBuffer vkCommandBuffer = VK_NULL_HANDLE;
            {
                VkCommandBufferAllocateInfo descCommandBufferAllocateInfo = {};
                descCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                descCommandBufferAllocateInfo.commandPool = m_owner->m_vkCommandPoolGraphics;
                descCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                descCommandBufferAllocateInfo.commandBufferCount = 1;
                TRYVK(vkAllocateCommandBuffers(m_owner->m_vkDevice, &descCommandBufferAllocateInfo, &vkCommandBuffer));
            }
            {
                VkCommandBufferBeginInfo descCommandBufferBeginInfo = {};
                descCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                descCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                TRYVK(vkBeginCommandBuffer(vkCommandBuffer, &descCommandBufferBeginInfo));
            }
            {
                VkImageMemoryBarrier descImageMemoryBarrier = {};
                descImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                descImageMemoryBarrier.srcAccessMask = 0;
                descImageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                descImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                descImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                descImageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                descImageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                descImageMemoryBarrier.image = m_vkImage;
                descImageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                descImageMemoryBarrier.subresourceRange.levelCount = 1;
                descImageMemoryBarrier.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(vkCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 1, &descImageMemoryBarrier);
            }
            {
                VkBufferImageCopy descBufferImageCopy = {};
                descBufferImageCopy.bufferOffset = 0;
                descBufferImageCopy.bufferRowLength = 0;
                descBufferImageCopy.bufferImageHeight = 0;
                descBufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                descBufferImageCopy.imageSubresource.mipLevel = 0;
                descBufferImageCopy.imageSubresource.baseArrayLayer = 0;
                descBufferImageCopy.imageSubresource.layerCount = 1;
                descBufferImageCopy.imageOffset = { 0, 0, 0 };
                descBufferImageCopy.imageExtent = { width, height, 1 };
                vkCmdCopyBufferToImage(vkCommandBuffer, vkStagingBuffer, m_vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &descBufferImageCopy);
            }
            {
                VkImageMemoryBarrier descImageMemoryBarrier = {};
                descImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                descImageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                descImageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                descImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                descImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                descImageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                descImageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                descImageMemoryBarrier.image = m_vkImage;
                descImageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                descImageMemoryBarrier.subresourceRange.levelCount = 1;
                descImageMemoryBarrier.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(vkCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 1, &descImageMemoryBarrier);
            }
            TRYVK(vkEndCommandBuffer(vkCommandBuffer));
            {
                VkSubmitInfo descSubmitInfo = {};
                descSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                descSubmitInfo.commandBufferCount = 1;
                descSubmitInfo.pCommandBuffers = &vkCommandBuffer;
                TRYVK(vkQueueSubmit(m_owner->m_vkQueueGraphics, 1, &descSubmitInfo, VK_NULL_HANDLE));
            }
            TRYVK(vkQueueWaitIdle(m_owner->m_vkQueueGraphics));
            // Clean up.
            vkFreeCommandBuffers(m_owner->m_vkDevice, m_owner->m_vkCommandPoolGraphics, 1, &vkCommandBuffer);
            vkCommandBuffer = VK_NULL_HANDLE;
            vkFreeMemory(owner->m_vkDevice, vkStagingBufferMemory, nullptr);
            vkStagingBufferMemory = VK_NULL_HANDLE;
            vkDestroyBuffer(owner->m_vkDevice, vkStagingBuffer, nullptr);
            vkStagingBuffer = VK_NULL_HANDLE;
            // Create an image view.
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
                descImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
                descImageViewCreateInfo.subresourceRange.levelCount = 1;
                descImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
                descImageViewCreateInfo.subresourceRange.layerCount = 1;
                TRYVK(vkCreateImageView(m_owner->m_vkDevice, &descImageViewCreateInfo, nullptr, &m_vkImageView));
            }
        }
    }
    ITexture_Vulkan::~ITexture_Vulkan()
    {
        vkDestroyImageView(m_owner->m_vkDevice, m_vkImageView, nullptr);
        m_vkImageView = VK_NULL_HANDLE;
        vkFreeMemory(m_owner->m_vkDevice, m_vkImageMemory, nullptr);
        m_vkImageMemory = VK_NULL_HANDLE;
        vkDestroyImage(m_owner->m_vkDevice, m_vkImage, nullptr);
        m_vkImage = VK_NULL_HANDLE;
    }
}