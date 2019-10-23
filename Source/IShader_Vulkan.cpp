#include "ErrorVK.h"
#include "IDevice3D_Vulkan.h"
#include "IShader_Vulkan.h"
#include "Vector.h"

#include <fstream>

namespace Arcturus
{
    IShader_Vulkan::~IShader_Vulkan()
    {
        vkDestroyPipeline(m_owner->m_vkDevice, m_vkPipeline, nullptr);
        m_vkPipeline = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_owner->m_vkDevice, m_vkShaderModuleFragment, nullptr);
        m_vkShaderModuleFragment = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_owner->m_vkDevice, m_vkShaderModuleVertex, nullptr);
        m_vkShaderModuleVertex = VK_NULL_HANDLE;
    }

    IShader_Vulkan::IShader_Vulkan(IDevice3D_Vulkan* owner) : m_owner(owner)
    {
        // Create a vertex shader.
        {
            VkShaderModuleCreateInfo descShaderModuleCreateInfo = {};
            descShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            std::ifstream datastream("VulkanVertex.spv", std::ios::ate | std::ios::binary);
            std::ifstream::pos_type length = datastream.tellg();
            std::unique_ptr<char[]> data(new char[length]);
            datastream.seekg(0, std::ios::beg);
            datastream.read(data.get(), length);
            datastream.close();
            descShaderModuleCreateInfo.codeSize = length;
            descShaderModuleCreateInfo.pCode = reinterpret_cast<uint32_t*>(data.get());
            TRYVK(vkCreateShaderModule(m_owner->m_vkDevice, &descShaderModuleCreateInfo, nullptr, &m_vkShaderModuleVertex));
        }
        // Create a fragment shader.
        {
            VkShaderModuleCreateInfo descShaderModuleCreateInfo = {};
            descShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            std::ifstream datastream("VulkanFragment.spv", std::ios::ate | std::ios::binary);
            std::ifstream::pos_type length = datastream.tellg();
            std::unique_ptr<char[]> data(new char[length]);
            datastream.seekg(0, std::ios::beg);
            datastream.read(data.get(), length);
            datastream.close();
            descShaderModuleCreateInfo.codeSize = length;
            descShaderModuleCreateInfo.pCode = reinterpret_cast<uint32_t*>(data.get());
            TRYVK(vkCreateShaderModule(m_owner->m_vkDevice, &descShaderModuleCreateInfo, nullptr, &m_vkShaderModuleFragment));
        }
        // Create a graphics pipeline (depends on vkShaderModules).
        {
            VkPipelineShaderStageCreateInfo descPipelineShaderStageCreateInfo[2] = {};
            descPipelineShaderStageCreateInfo[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            descPipelineShaderStageCreateInfo[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            descPipelineShaderStageCreateInfo[0].module = m_vkShaderModuleVertex;
            descPipelineShaderStageCreateInfo[0].pName = "main";
            descPipelineShaderStageCreateInfo[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            descPipelineShaderStageCreateInfo[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            descPipelineShaderStageCreateInfo[1].module = m_vkShaderModuleFragment;
            descPipelineShaderStageCreateInfo[1].pName = "main";
            VkVertexInputBindingDescription descVertexInputBindingDescription = {};
            descVertexInputBindingDescription.binding = 0;
            descVertexInputBindingDescription.stride = sizeof(Vertex);
            descVertexInputBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            VkVertexInputAttributeDescription descVertexInputAttributeDescription[2] = {};
            descVertexInputAttributeDescription[0].location = 0;
            descVertexInputAttributeDescription[0].format = VK_FORMAT_R32G32_SFLOAT;
            descVertexInputAttributeDescription[0].offset = offsetof(Vertex, Position);
            descVertexInputAttributeDescription[1].location = 1;
            descVertexInputAttributeDescription[1].format = VK_FORMAT_B8G8R8A8_UNORM;
            descVertexInputAttributeDescription[1].offset = offsetof(Vertex, Color);
            VkPipelineVertexInputStateCreateInfo descPipelineVertexInputStateCreateInfo = {};
            descPipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            descPipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
            descPipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = &descVertexInputBindingDescription;
            descPipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = _countof(descVertexInputAttributeDescription);
            descPipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = descVertexInputAttributeDescription;
            VkPipelineInputAssemblyStateCreateInfo descPipelineInputAssemblyStateCreateInfo = {};
            descPipelineInputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            descPipelineInputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkViewport descViewport = {};
            VkRect2D descRect2DScissor = {};
            VkPipelineViewportStateCreateInfo descPipelineViewportStateCreateInfo = {};
            descPipelineViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            descPipelineViewportStateCreateInfo.viewportCount = 1;
            descPipelineViewportStateCreateInfo.pViewports = &descViewport;
            descPipelineViewportStateCreateInfo.scissorCount = 1;
            descPipelineViewportStateCreateInfo.pScissors = &descRect2DScissor;
            VkPipelineRasterizationStateCreateInfo descPipelineRasterizationStateCreateInfo = {};
            descPipelineRasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            descPipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
            descPipelineRasterizationStateCreateInfo.lineWidth = 1;
            descPipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_NONE;
            descPipelineRasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
            VkPipelineMultisampleStateCreateInfo descPipelineMultisampleStateCreateInfo = {};
            descPipelineMultisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            descPipelineMultisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState descPipelineColorBlendAttachmentState = {};
            descPipelineColorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            descPipelineColorBlendAttachmentState.blendEnable = VK_TRUE;
            descPipelineColorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            descPipelineColorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            descPipelineColorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
            descPipelineColorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            descPipelineColorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            descPipelineColorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
            VkPipelineColorBlendStateCreateInfo descPipelineColorBlendStateCreateInfo = {};
            descPipelineColorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            descPipelineColorBlendStateCreateInfo.attachmentCount = 1;
            descPipelineColorBlendStateCreateInfo.pAttachments = &descPipelineColorBlendAttachmentState;
            VkPipelineDynamicStateCreateInfo descPipelineDynamicStateCreateInfo = {};
            descPipelineDynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            VkDynamicState descDynamicState[2] = {};
            descDynamicState[0] = VK_DYNAMIC_STATE_VIEWPORT;
            descDynamicState[1] = VK_DYNAMIC_STATE_SCISSOR;
            descPipelineDynamicStateCreateInfo.dynamicStateCount = _countof(descDynamicState);
            descPipelineDynamicStateCreateInfo.pDynamicStates = descDynamicState;
            VkGraphicsPipelineCreateInfo descGraphicsPipelineCreateInfo = {};
            descGraphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            descGraphicsPipelineCreateInfo.stageCount = _countof(descPipelineShaderStageCreateInfo);
            descGraphicsPipelineCreateInfo.pStages = descPipelineShaderStageCreateInfo;
            descGraphicsPipelineCreateInfo.pVertexInputState = &descPipelineVertexInputStateCreateInfo;
            descGraphicsPipelineCreateInfo.pInputAssemblyState = &descPipelineInputAssemblyStateCreateInfo;
            descGraphicsPipelineCreateInfo.pViewportState = &descPipelineViewportStateCreateInfo;
            descGraphicsPipelineCreateInfo.pRasterizationState = &descPipelineRasterizationStateCreateInfo;
            descGraphicsPipelineCreateInfo.pMultisampleState = &descPipelineMultisampleStateCreateInfo;
            descGraphicsPipelineCreateInfo.pColorBlendState = &descPipelineColorBlendStateCreateInfo;
            descGraphicsPipelineCreateInfo.pDynamicState = &descPipelineDynamicStateCreateInfo;
            descGraphicsPipelineCreateInfo.layout = m_owner->m_vkPipelineLayoutEmpty;
            descGraphicsPipelineCreateInfo.renderPass = m_owner->m_vkRenderPass;
            TRYVK(vkCreateGraphicsPipelines(m_owner->m_vkDevice, m_owner->m_vkPipelineCache, 1, &descGraphicsPipelineCreateInfo, nullptr, &m_vkPipeline)); 
        }
    }
}