#include "IConstantBuffer_Vulkan.h"
#include "IConstantBufferView_Vulkan.h"
#include "IDevice3D_Vulkan.h"
#include "VulkanInclude.h"

namespace Arcturus
{
    IConstantBufferView_Vulkan::IConstantBufferView_Vulkan(IDevice3D_Vulkan* owner, IConstantBuffer_Vulkan* constantBuffer) : m_owner(owner)
    {
        VkDescriptorBufferInfo descDescriptorBufferInfo = {};
        descDescriptorBufferInfo.buffer = constantBuffer->m_vkBuffer;
        descDescriptorBufferInfo.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet descWriteDescriptorSet = {};
        descWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descWriteDescriptorSet.dstSet = m_owner->m_vkDescriptorSetUAV;
        descWriteDescriptorSet.dstBinding = 0;
        descWriteDescriptorSet.descriptorCount = 1;
        descWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descWriteDescriptorSet.pBufferInfo = &descDescriptorBufferInfo;
        vkUpdateDescriptorSets(m_owner->m_vkDevice, 1, &descWriteDescriptorSet, 0, nullptr);
    }
}