#include "ErrorVK.h"
#include "IBuffer_Vulkan.h"
#include "IDevice3D_Vulkan.h"

namespace Arcturus
{
    IBuffer_Vulkan::IBuffer_Vulkan(IDevice3D_Vulkan* owner, uint32_t dataSize, const void* data, VkBufferUsageFlags usage) : m_owner(owner), m_byteSize(dataSize)
    {
        VkBufferCreateInfo descBuffer = {};
        descBuffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        descBuffer.size = m_byteSize;
        descBuffer.usage = usage;
        descBuffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        TRYVK(vkCreateBuffer(m_owner->m_vkDevice, &descBuffer, nullptr, &m_vkBuffer));
        VkMemoryRequirements descMemoryReq = {};
        vkGetBufferMemoryRequirements(m_owner->m_vkDevice, m_vkBuffer, &descMemoryReq);
        VkMemoryAllocateInfo descAlloc = {};
        descAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        descAlloc.allocationSize = descMemoryReq.size;
        descAlloc.memoryTypeIndex = m_owner->m_memoryTypeShared;
        TRYVK(vkAllocateMemory(m_owner->m_vkDevice, &descAlloc, nullptr, &m_vkDeviceMemory));
        TRYVK(vkBindBufferMemory(m_owner->m_vkDevice, m_vkBuffer, m_vkDeviceMemory, 0));
        void *hostPtr = nullptr;
        TRYVK(vkMapMemory(m_owner->m_vkDevice, m_vkDeviceMemory, 0, descMemoryReq.size, 0, &hostPtr));
        memcpy(hostPtr, data, m_byteSize);
        vkUnmapMemory(m_owner->m_vkDevice, m_vkDeviceMemory);
    }

    IBuffer_Vulkan::~IBuffer_Vulkan()
    {
        vkDestroyBuffer(m_owner->m_vkDevice, m_vkBuffer, nullptr);
        m_vkBuffer = VK_NULL_HANDLE;
        vkFreeMemory(m_owner->m_vkDevice, m_vkDeviceMemory, nullptr);
        m_vkDeviceMemory = VK_NULL_HANDLE;
    }
}