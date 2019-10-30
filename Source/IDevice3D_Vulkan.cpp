#include "ErrorVK.h"
#include "IConstantBuffer_Vulkan.h"
#include "IConstantBufferView_Vulkan.h"
#include "IDevice3D_Vulkan.h"
#include "IIndexBuffer_Vulkan.h"
#include "IRenderTarget_Vulkan.h"
#include "IShader_Vulkan.h"
#include "ITexture_Vulkan.h"
#include "IVertexBuffer_Vulkan.h"

namespace Arcturus
{
    IDevice3D_Vulkan::IDevice3D_Vulkan() : m_vkCommandBuffer(VK_NULL_HANDLE)
    {
        ////////////////////////////////////////////////////////////////////////////////
        // Create a Vulkan instance.
        m_vkInstance = VK_NULL_HANDLE;
        {
            VkApplicationInfo descAppInfo = {};
            descAppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            descAppInfo.pApplicationName = "DrawingContextVulkan";
            descAppInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            descAppInfo.pEngineName = "Vulkan 1.0";
            descAppInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            descAppInfo.apiVersion = VK_API_VERSION_1_0;
            VkInstanceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &descAppInfo;
#ifndef NDEBUG
            const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
            createInfo.ppEnabledLayerNames = validationLayers.data();
            createInfo.enabledLayerCount = 1;
#endif
            TRYVK(vkCreateInstance(&createInfo, nullptr, &m_vkInstance));
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Get a list of all Vulkan devices.
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        {
            uint32_t countDevices = 0;
            TRYVK(vkEnumeratePhysicalDevices(m_vkInstance, &countDevices, nullptr));
            std::vector<VkPhysicalDevice> descDevices(countDevices);
            TRYVK(vkEnumeratePhysicalDevices(m_vkInstance, &countDevices, descDevices.data()));
            // Find a suitable hardware device.
            for (const auto& device : descDevices)
            {
                VkPhysicalDeviceProperties deviceProperties;
                vkGetPhysicalDeviceProperties(device, &deviceProperties);
                if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                {
                    physicalDevice = device;
                    break;
                }
            }
            if (physicalDevice == VK_NULL_HANDLE)
            {
                throw std::exception("Unable to find a compatible device.");
            }
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Find the graphics queue.
        m_queueGraphics = -1;
        {
            uint32_t countQueueFamilies = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &countQueueFamilies, nullptr);
            std::vector<VkQueueFamilyProperties> descQueueFamilies(countQueueFamilies);
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &countQueueFamilies, descQueueFamilies.data());
            // Find the graphics queue.
            for (int i = 0; i < countQueueFamilies; ++i)
            {
                if (descQueueFamilies[i].queueCount > 0 && descQueueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    m_queueGraphics = i;
                    break;
                }
            }
        }
        if (m_queueGraphics == -1)
        {
            throw std::exception("Unable to find a graphics queue.");
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Locate the host and device memory heaps and types.
        m_memoryHeapHost = -1;
        m_memoryHeapDevice = -1;
        {
            VkPhysicalDeviceMemoryProperties descMemoryProps = {};
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &descMemoryProps);
            // Locate the host and device memory heaps.
            for (int i = 0; i < descMemoryProps.memoryHeapCount; ++i)
            {
                if (m_memoryHeapHost == -1 && descMemoryProps.memoryHeaps[i].flags == 0)
                {
                    m_memoryHeapHost = i;
                }
                if (m_memoryHeapDevice == -1 && descMemoryProps.memoryHeaps[i].flags == VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                {
                    m_memoryHeapDevice = i;
                }
            }
            if (m_memoryHeapHost == -1 || m_memoryHeapDevice == -1)
            {
                throw std::exception("Unable to find all required Vulkan device heaps.");
            }
            // Locate the host and device memory types.
            m_memoryTypeDevice = -1;
            m_memoryTypeShared = -1;
            for (int i = 0; i < descMemoryProps.memoryTypeCount; ++i)
            {
                if (m_memoryTypeDevice == -1 && descMemoryProps.memoryTypes[i].heapIndex == m_memoryHeapDevice && descMemoryProps.memoryTypes[i].propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                {
                    m_memoryTypeDevice = i;
                }
                if (m_memoryTypeShared == -1 && descMemoryProps.memoryTypes[i].heapIndex == m_memoryHeapHost && descMemoryProps.memoryTypes[i].propertyFlags == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                {
                    m_memoryTypeShared = i;
                }
            }
            if (m_memoryTypeDevice == -1 || m_memoryTypeShared == -1)
            {
                throw std::exception("Unable to find all required Vulkan device memory types.");
            }
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Create the device.
        m_vkDevice = VK_NULL_HANDLE;
        {
            // Define the queues we will require.
            VkDeviceQueueCreateInfo descQueueCreateInfo = {};
            descQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            descQueueCreateInfo.queueFamilyIndex = m_queueGraphics;
            descQueueCreateInfo.queueCount = 1;
            float queuePriority = 1;
            descQueueCreateInfo.pQueuePriorities = &queuePriority;
            // Find all extensions; we're going to enable them all.
            uint32_t extensionCount = 0;
            TRYVK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr));
            std::vector<VkExtensionProperties> extensions(extensionCount);
            TRYVK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data()));
            // Create the device.
            VkDeviceCreateInfo descDeviceCreateInfo = {};
            descDeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            descDeviceCreateInfo.pQueueCreateInfos = &descQueueCreateInfo;
            descDeviceCreateInfo.queueCreateInfoCount = 1;
            descDeviceCreateInfo.enabledExtensionCount = extensionCount;
            // Attach all the extensions from above.
            std::vector<const char*> extensionNames;
            for (const auto& ext : extensions)
            {
                extensionNames.push_back(ext.extensionName);
            }
            descDeviceCreateInfo.ppEnabledExtensionNames = extensionNames.data();
            TRYVK(vkCreateDevice(physicalDevice, &descDeviceCreateInfo, nullptr, &m_vkDevice));
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Create a command queue.
        vkGetDeviceQueue(m_vkDevice, m_queueGraphics, 0, &m_vkQueueGraphics);
        ////////////////////////////////////////////////////////////////////////////////
        // Initialize the dynamic extension loader.
        m_dynamicLoader.init(m_vkInstance, m_vkDevice);
        ////////////////////////////////////////////////////////////////////////////////
        // Create a command buffer pool.
        {
            VkCommandPoolCreateInfo descCommandPool = {};
            descCommandPool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            descCommandPool.queueFamilyIndex = m_queueGraphics;
            TRYVK(vkCreateCommandPool(m_vkDevice, &descCommandPool, nullptr, &m_vkCommandPoolGraphics));
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Create the descriptor pool and layouts(s).
        {
            VkDescriptorPoolSize descDescriptorPoolSize[3];
            descDescriptorPoolSize[0].descriptorCount = 64;
            descDescriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descDescriptorPoolSize[1].descriptorCount = 64;
            descDescriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            descDescriptorPoolSize[2].descriptorCount = 64;
            descDescriptorPoolSize[2].type = VK_DESCRIPTOR_TYPE_SAMPLER;
            VkDescriptorPoolCreateInfo descDescriptorPoolCreateInfo = {};
            descDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descDescriptorPoolCreateInfo.maxSets = 64;
            descDescriptorPoolCreateInfo.poolSizeCount = _countof(descDescriptorPoolSize);
            descDescriptorPoolCreateInfo.pPoolSizes = descDescriptorPoolSize;
            TRYVK(vkCreateDescriptorPool(m_vkDevice, &descDescriptorPoolCreateInfo, nullptr, &m_vkDescriptorPool));
        }
        {
            VkDescriptorSetLayoutBinding descDescriptorSetLayoutBinding = {};
            descDescriptorSetLayoutBinding.binding = 0;
            descDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descDescriptorSetLayoutBinding.descriptorCount = 1;
            descDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            VkDescriptorSetLayoutCreateInfo descDescriptorSetLayoutCreateInfo = {};
            descDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descDescriptorSetLayoutCreateInfo.bindingCount = 1;
            descDescriptorSetLayoutCreateInfo.pBindings = &descDescriptorSetLayoutBinding;
            TRYVK(vkCreateDescriptorSetLayout(m_vkDevice, &descDescriptorSetLayoutCreateInfo, nullptr, &m_vkDescriptorSetLayoutUAV));
        }
        {
            VkDescriptorSetAllocateInfo descDescriptorSetAllocateInfo = {};
            descDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descDescriptorSetAllocateInfo.descriptorPool = m_vkDescriptorPool;
            descDescriptorSetAllocateInfo.descriptorSetCount = 1;
            descDescriptorSetAllocateInfo.pSetLayouts = &m_vkDescriptorSetLayoutUAV;
            TRYVK(vkAllocateDescriptorSets(m_vkDevice, &descDescriptorSetAllocateInfo, &m_vkDescriptorSetUAV));
        }
        {
            VkDescriptorSetLayoutBinding descDescriptorSetLayoutBinding = {};
            descDescriptorSetLayoutBinding.binding = 0;
            descDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            descDescriptorSetLayoutBinding.descriptorCount = 1;
            descDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo descDescriptorSetLayoutCreateInfo = {};
            descDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descDescriptorSetLayoutCreateInfo.bindingCount = 1;
            descDescriptorSetLayoutCreateInfo.pBindings = &descDescriptorSetLayoutBinding;
            TRYVK(vkCreateDescriptorSetLayout(m_vkDevice, &descDescriptorSetLayoutCreateInfo, nullptr, &m_vkDescriptorSetLayoutSRV));
        }
        {
            VkDescriptorSetAllocateInfo descDescriptorSetAllocateInfo = {};
            descDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descDescriptorSetAllocateInfo.descriptorPool = m_vkDescriptorPool;
            descDescriptorSetAllocateInfo.descriptorSetCount = 1;
            descDescriptorSetAllocateInfo.pSetLayouts = &m_vkDescriptorSetLayoutSRV;
            TRYVK(vkAllocateDescriptorSets(m_vkDevice, &descDescriptorSetAllocateInfo, &m_vkDescriptorSetSRV));
        }
        {
            VkDescriptorSetLayoutBinding descDescriptorSetLayoutBinding = {};
            descDescriptorSetLayoutBinding.binding = 0;
            descDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            descDescriptorSetLayoutBinding.descriptorCount = 1;
            descDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo descDescriptorSetLayoutCreateInfo = {};
            descDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descDescriptorSetLayoutCreateInfo.bindingCount = 1;
            descDescriptorSetLayoutCreateInfo.pBindings = &descDescriptorSetLayoutBinding;
            TRYVK(vkCreateDescriptorSetLayout(m_vkDevice, &descDescriptorSetLayoutCreateInfo, nullptr, &m_vkDescriptorSetLayoutSMP));
        }
        {
            VkDescriptorSetAllocateInfo descDescriptorSetAllocateInfo = {};
            descDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descDescriptorSetAllocateInfo.descriptorPool = m_vkDescriptorPool;
            descDescriptorSetAllocateInfo.descriptorSetCount = 1;
            descDescriptorSetAllocateInfo.pSetLayouts = &m_vkDescriptorSetLayoutSMP;
            TRYVK(vkAllocateDescriptorSets(m_vkDevice, &descDescriptorSetAllocateInfo, &m_vkDescriptorSetSMP));
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Create a pipeline cache.
        {
            VkPipelineCacheCreateInfo descPipelineCacheCreateInfo = {};
            descPipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
            TRYVK(vkCreatePipelineCache(m_vkDevice, &descPipelineCacheCreateInfo, nullptr, &m_vkPipelineCache));
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Create a pipeline layout.
        {
            VkDescriptorSetLayout allLayouts[3] = { m_vkDescriptorSetLayoutUAV, m_vkDescriptorSetLayoutSRV, m_vkDescriptorSetLayoutSMP };
            VkPipelineLayoutCreateInfo descPipelineLayout = {};
            descPipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            descPipelineLayout.setLayoutCount = 3;
            descPipelineLayout.pSetLayouts = allLayouts;
            TRYVK(vkCreatePipelineLayout(m_vkDevice, &descPipelineLayout, nullptr, &m_vkPipelineLayoutEmpty));
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Create a render pass.
        {
            VkAttachmentDescription descAttachmentDescription = {};
            descAttachmentDescription.format = VK_FORMAT_R8G8B8A8_UNORM;
            descAttachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
            descAttachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            descAttachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            descAttachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            descAttachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            descAttachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            descAttachmentDescription.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference descAttachmentReference = {};
            descAttachmentReference.attachment = 0;
            descAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkSubpassDescription descSubpassDescription = {};
            descSubpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            descSubpassDescription.colorAttachmentCount = 1;
            descSubpassDescription.pColorAttachments = &descAttachmentReference;
            VkRenderPassCreateInfo descRenderPassCreateInfo = {};
            descRenderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            descRenderPassCreateInfo.attachmentCount = 1;
            descRenderPassCreateInfo.pAttachments = &descAttachmentDescription;
            descRenderPassCreateInfo.subpassCount = 1;
            descRenderPassCreateInfo.pSubpasses = &descSubpassDescription;
            TRYVK(vkCreateRenderPass(m_vkDevice, &descRenderPassCreateInfo, nullptr, &m_vkRenderPass));
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Create a sampler.
        {
            VkSamplerCreateInfo descSamplerCreateInfo = {};
            descSamplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            descSamplerCreateInfo.magFilter = VK_FILTER_LINEAR;
            descSamplerCreateInfo.minFilter = VK_FILTER_LINEAR;
            descSamplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            descSamplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            descSamplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            descSamplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            TRYVK(vkCreateSampler(m_vkDevice, &descSamplerCreateInfo, nullptr, &m_vkSampler));
        }
        ////////////////////////////////////////////////////////////////////////////////
        // Try some raytracing.
        //
        // Eventually. This is just the same junk as DX12 with slightly different names.
        /*
        VkAccelerationStructureNV vkAccelerationStructure = VK_NULL_HANDLE;
        {
            VkGeometryNV descGeometries[1] = {};
            descGeometries[0].sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
            descGeometries[0].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
            descGeometries[0].geometry.triangles.vertexCount = 3;
            descGeometries[0].geometry.triangles.vertexStride = 12;
            descGeometries[0].geometry.triangles.vertexFormat = VK_FORMAT_R32G32_SFLOAT;
            descGeometries[0].geometry.triangles.indexCount = 3;
            descGeometries[0].geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
            VkAccelerationStructureCreateInfoNV descAccelerationStructureCreateInfo = {};
            descAccelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
            descAccelerationStructureCreateInfo.info.geometryCount = _countof(descGeometries);
            descAccelerationStructureCreateInfo.info.pGeometries = descGeometries;
            descAccelerationStructureCreateInfo.info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
            TRYVK(m_dynamicLoader.vkCreateAccelerationStructureNV(m_vkDevice, &descAccelerationStructureCreateInfo, nullptr, &vkAccelerationStructure));
        }
        VkMemoryRequirements2KHR descMemoryRequirements = {};
        {
            VkAccelerationStructureMemoryRequirementsInfoNV descAccelerationStructureMemoryRequirementsInfoNV = {};
            descAccelerationStructureMemoryRequirementsInfoNV.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
            descAccelerationStructureMemoryRequirementsInfoNV.accelerationStructure = vkAccelerationStructure;
            m_dynamicLoader.vkGetAccelerationStructureMemoryRequirementsNV(m_vkDevice, &descAccelerationStructureMemoryRequirementsInfoNV, &descMemoryRequirements);
        }
        */
    }

    IDevice3D_Vulkan::~IDevice3D_Vulkan()
    {
        // Destroy the sampler.
        vkDestroySampler(m_vkDevice, m_vkSampler, nullptr);
        m_vkSampler = VK_NULL_HANDLE;
        // Destroy the render pass.
        vkDestroyRenderPass(m_vkDevice, m_vkRenderPass, nullptr);
        m_vkRenderPass = VK_NULL_HANDLE;
        // Destroy the pipeline layout.
        vkDestroyPipelineLayout(m_vkDevice, m_vkPipelineLayoutEmpty, nullptr);
        m_vkPipelineLayoutEmpty = VK_NULL_HANDLE;
        // Destroy the pipeline cache.
        vkDestroyPipelineCache(m_vkDevice, m_vkPipelineCache, nullptr);
        m_vkPipelineCache = VK_NULL_HANDLE;
        // Destroy the descriptor layout(s).
        vkDestroyDescriptorSetLayout(m_vkDevice, m_vkDescriptorSetLayoutSMP, nullptr);
        m_vkDescriptorSetLayoutSMP = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_vkDevice, m_vkDescriptorSetLayoutSRV, nullptr);
        m_vkDescriptorSetLayoutSRV = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_vkDevice, m_vkDescriptorSetLayoutUAV, nullptr);
        m_vkDescriptorSetLayoutUAV = VK_NULL_HANDLE;
        // Destroy the descriptor pool.
        vkDestroyDescriptorPool(m_vkDevice, m_vkDescriptorPool, nullptr);
        m_vkDescriptorPool = VK_NULL_HANDLE;
        // Destroy the command pool.
        vkDestroyCommandPool(m_vkDevice, m_vkCommandPoolGraphics, nullptr);
        m_vkCommandPoolGraphics = VK_NULL_HANDLE;
        // Unreference the queue (we did not allocate this).
        m_vkQueueGraphics = VK_NULL_HANDLE;
        // Destroy the Vulkan device.
        vkDestroyDevice(m_vkDevice, nullptr);
        m_vkDevice = VK_NULL_HANDLE;
        // Destroy the Vulkan instance.
        vkDestroyInstance(m_vkInstance, nullptr);
        m_vkInstance = nullptr;
    }

    IConstantBuffer* IDevice3D_Vulkan::CreateConstantBuffer(uint32_t dataSize, const void* data)
    {
        return new IConstantBuffer_Vulkan(this, dataSize, data);
    }

    IConstantBufferView* IDevice3D_Vulkan::CreateConstantBufferView(IConstantBuffer* constantBuffer)
    {
        return new IConstantBufferView_Vulkan(this, dynamic_cast<IConstantBuffer_Vulkan*>(constantBuffer));
    }

    IIndexBuffer* IDevice3D_Vulkan::CreateIndexBuffer(uint32_t dataSize, const void* data)
    {
        return new IIndexBuffer_Vulkan(this, dataSize, data);
    }

    IRenderTarget* IDevice3D_Vulkan::CreateRenderTarget(const RenderTargetDeclaration& declaration)
    {
        throw std::exception("Not Implemented.");
    }

    IShader* IDevice3D_Vulkan::CreateShader()
    {
        return new IShader_Vulkan(this);
    }

    ITexture* IDevice3D_Vulkan::CreateTexture2D(uint32_t width, uint32_t height, const void* data)
    {
        return new ITexture_Vulkan(this, width, height, data);
    }

    IVertexBuffer* IDevice3D_Vulkan::CreateVertexBuffer(uint32_t dataSize, uint32_t strideSize, const void* data)
    {
        return new IVertexBuffer_Vulkan(this, dataSize, data);
    }

    IRenderTarget* IDevice3D_Vulkan::OpenRenderTarget(const RenderTargetDeclaration& declaration, HANDLE handle)
    {
        return new IRenderTarget_Vulkan(this, declaration, handle);
    }

    void IDevice3D_Vulkan::CopyResource(IRenderTarget* destination, IRenderTarget* source)
    {
        throw std::exception("Not Implemented.");
    }

    // TODO: Context calls - these will need to be moved later.
    void IDevice3D_Vulkan::BeginRender()
    {
        assert(m_vkCommandBuffer == VK_NULL_HANDLE);
        // Create a command buffer.
        {
            VkCommandBufferAllocateInfo descCommandBufferAllocateInfo = {};
            descCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            descCommandBufferAllocateInfo.commandPool = m_vkCommandPoolGraphics;
            descCommandBufferAllocateInfo.commandBufferCount = 1;
            descCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            TRYVK(vkAllocateCommandBuffers(m_vkDevice, &descCommandBufferAllocateInfo, &m_vkCommandBuffer));
        }
        // Create a command buffer and start filling it.
        {
            VkCommandBufferBeginInfo descCommandBufferBeginInfo = {};
            descCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            descCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            TRYVK(vkBeginCommandBuffer(m_vkCommandBuffer, &descCommandBufferBeginInfo));
        }
    }

    void IDevice3D_Vulkan::EndRender()
    {
        assert(m_vkCommandBuffer != VK_NULL_HANDLE);
        // Finish up command buffer generation.
        TRYVK(vkEndCommandBuffer(m_vkCommandBuffer));
        // Submit the command buffer to GPU.
        {
            VkSubmitInfo descSubmitInfo = {};
            descSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            descSubmitInfo.commandBufferCount = 1;
            descSubmitInfo.pCommandBuffers = &m_vkCommandBuffer;
            TRYVK(vkQueueSubmit(m_vkQueueGraphics, 1, &descSubmitInfo, nullptr));
        }
        // Wait until the whole queue has been executed and the GPU goes idle.
        TRYVK(vkQueueWaitIdle(m_vkQueueGraphics));
        // Cleanup.
        vkFreeCommandBuffers(m_vkDevice, m_vkCommandPoolGraphics, 1, &m_vkCommandBuffer);
        m_vkCommandBuffer = VK_NULL_HANDLE;
    }

    void IDevice3D_Vulkan::BeginPass(IRenderTarget* renderTarget, const Color& clearColor)
    {
        IRenderTarget_Vulkan* renderTargetVK = dynamic_cast<IRenderTarget_Vulkan*>(renderTarget);
        VkRenderPassBeginInfo descRenderPassBeginInfo = {};
        descRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        descRenderPassBeginInfo.renderPass = m_vkRenderPass;
        descRenderPassBeginInfo.framebuffer = renderTargetVK->m_vkFramebuffer;
        descRenderPassBeginInfo.renderArea.offset = { 0, 0 };
        descRenderPassBeginInfo.renderArea.extent = { renderTargetVK->m_width, renderTargetVK->m_height };
        VkClearValue descClearValue = { 0, 0, 0, 0 };
        descRenderPassBeginInfo.clearValueCount = 1;
        descRenderPassBeginInfo.pClearValues = &descClearValue;
        vkCmdBeginRenderPass(m_vkCommandBuffer, &descRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    }
    
    void IDevice3D_Vulkan::EndPass()
    {
        vkCmdEndRenderPass(m_vkCommandBuffer);
    }

    void IDevice3D_Vulkan::SetShader(IShader* shader)
    {
        vkCmdBindPipeline(m_vkCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, dynamic_cast<IShader_Vulkan*>(shader)->m_vkPipeline);
    }

    void IDevice3D_Vulkan::SetTexture(ITexture* texture)
    {
        ITexture_Vulkan* textureVK = dynamic_cast<ITexture_Vulkan*>(texture);
        {
            VkDescriptorImageInfo descDescriptorImageInfo = {};
            descDescriptorImageInfo.sampler = m_vkSampler;
            descDescriptorImageInfo.imageView = textureVK->m_vkImageView;
            descDescriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet descWriteDescriptorSet = {};
            descWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descWriteDescriptorSet.dstSet = m_vkDescriptorSetSRV;
            descWriteDescriptorSet.dstBinding = 0;
            descWriteDescriptorSet.descriptorCount = 1;
            descWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            descWriteDescriptorSet.pImageInfo = &descDescriptorImageInfo;
            vkUpdateDescriptorSets(m_vkDevice, 1, &descWriteDescriptorSet, 0, nullptr);
        }
        {
            VkDescriptorImageInfo descDescriptorImageInfo = {};
            descDescriptorImageInfo.sampler = m_vkSampler;
            descDescriptorImageInfo.imageView = textureVK->m_vkImageView;
            descDescriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkWriteDescriptorSet descWriteDescriptorSet = {};
            descWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descWriteDescriptorSet.dstSet = m_vkDescriptorSetSMP;
            descWriteDescriptorSet.dstBinding = 0;
            descWriteDescriptorSet.descriptorCount = 1;
            descWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            descWriteDescriptorSet.pImageInfo = &descDescriptorImageInfo;
            vkUpdateDescriptorSets(m_vkDevice, 1, &descWriteDescriptorSet, 0, nullptr);
        }
        vkCmdBindDescriptorSets(m_vkCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vkPipelineLayoutEmpty, 1, 1, &m_vkDescriptorSetSRV, 0, nullptr);
        vkCmdBindDescriptorSets(m_vkCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vkPipelineLayoutEmpty, 2, 1, &m_vkDescriptorSetSMP, 0, nullptr);
    }

    void IDevice3D_Vulkan::SetViewport(const Viewport& viewport)
    {
        {
            VkViewport descViewport = {};
            descViewport.x = viewport.x;
            descViewport.y = viewport.y;
            descViewport.width = viewport.width;
            descViewport.height = viewport.height;
            descViewport.minDepth = viewport.minDepth;
            descViewport.maxDepth = viewport.maxDepth;
            vkCmdSetViewport(m_vkCommandBuffer, 0, 1, &descViewport);
        }
        {
            VkRect2D descRect2DScissor = {};
            descRect2DScissor.offset = { 0, 0 };
            descRect2DScissor.extent = { static_cast<uint32_t>(viewport.width), static_cast<uint32_t>(viewport.height) };
            vkCmdSetScissor(m_vkCommandBuffer, 0, 1, &descRect2DScissor);
        }
    }

    void IDevice3D_Vulkan::SetVertexBuffer(IVertexBuffer* vertexBuffer, uint32_t stride)
    {
        VkDeviceSize descOffsets = 0;
        vkCmdBindVertexBuffers(m_vkCommandBuffer, 0, 1, &dynamic_cast<IVertexBuffer_Vulkan*>(vertexBuffer)->m_vkBuffer, &descOffsets);
    }

    void IDevice3D_Vulkan::SetIndexBuffer(IIndexBuffer* indexBuffer)
    {
        vkCmdBindIndexBuffer(m_vkCommandBuffer, dynamic_cast<IIndexBuffer_Vulkan*>(indexBuffer)->m_vkBuffer, 0, VK_INDEX_TYPE_UINT32);
    }

    void IDevice3D_Vulkan::DrawIndexedPrimitives(uint32_t vertexCount, uint32_t indexCount)
    {
        vkCmdBindDescriptorSets(m_vkCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vkPipelineLayoutEmpty, 0, 1, &m_vkDescriptorSetUAV, 0, nullptr);
        vkCmdDrawIndexed(m_vkCommandBuffer, indexCount, 1, 0, 0, 0);
    }
}