#pragma once

#include "AutoRelease.h"
#include "IDevice3D.h"
#include "VulkanInclude.h"

#include <memory>

namespace Arcturus
{
    class IConstantBuffer_Vulkan;

    class IDevice3D_Vulkan : public IDevice3D
    {
    public:
        IDevice3D_Vulkan();
        ~IDevice3D_Vulkan();
        IConstantBuffer* CreateConstantBuffer(uint32_t dataSize, const void* data) override;
        IConstantBufferView* CreateConstantBufferView(IConstantBuffer* constantBuffer) override;
        IIndexBuffer* CreateIndexBuffer(uint32_t dataSize, const void* data) override;
        IRenderTarget* CreateRenderTarget(const RenderTargetDeclaration& declaration) override;
        IShader* CreateShader() override;
        IVertexBuffer* CreateVertexBuffer(uint32_t dataSize, uint32_t strideSize, const void* data) override;
        IRenderTarget* OpenRenderTarget(const RenderTargetDeclaration& declaration, HANDLE handle) override;
        void CopyResource(IRenderTarget* destination, IRenderTarget* source) override;
        // TODO: Context calls - these will need to be moved later.
        void BeginRender() override;
        void EndRender() override;
        void BeginPass(IRenderTarget* renderTarget, const Color& clearColor) override;
        void EndPass() override;
        void SetShader(IShader* shader) override;
        void SetViewport(const Viewport& viewport) override;
        void SetVertexBuffer(IVertexBuffer* vertexBuffer, uint32_t stride) override;
        void SetIndexBuffer(IIndexBuffer* indexBuffer) override;
        void DrawIndexedPrimitives(uint32_t vertexCount, uint32_t indexCount) override;
        VkInstance                              m_vkInstance;
        VkDevice                                m_vkDevice;
        vk::DispatchLoaderDynamic               m_dynamicLoader;
        VkQueue                                 m_vkQueueGraphics;
        uint32_t                                m_memoryHeapHost;
        uint32_t                                m_memoryHeapDevice;
        uint32_t                                m_memoryTypeDevice;
        uint32_t                                m_memoryTypeShared;
        uint32_t                                m_queueGraphics;
        VkCommandPool                           m_vkCommandPoolGraphics;
        VkDescriptorPool                        m_vkDescriptorPool;
        VkDescriptorSetLayout                   m_vkDescriptorSetLayout;
        VkDescriptorSet                         m_vkDescriptorSet;
        VkPipelineCache                         m_vkPipelineCache;
        VkPipelineLayout                        m_vkPipelineLayoutEmpty;
        VkRenderPass                            m_vkRenderPass;
        VkCommandBuffer                         m_vkCommandBuffer;
    };
}