#pragma once

#include "IDevice3D.h"
#include <atlbase.h>
#include <d3d12.h>
#include <memory>

namespace Arcturus
{
    class IDevice3D_D3D12 : public IDevice3D
    {
    public:
        IDevice3D_D3D12();
        IConstantBuffer* CreateConstantBuffer(uint32_t dataSize, const void* data) override;
        IConstantBufferView* CreateConstantBufferView(IConstantBuffer* constantBuffer) override;
        IIndexBuffer* CreateIndexBuffer(uint32_t dataSize, const void* data) override;
        IRenderTarget* CreateRenderTarget(const RenderTargetDeclaration& declaration) override;
        IShader* CreateShader() override;
        ITexture* CreateTexture2D(uint32_t width, uint32_t height, const void* data) override;
        IVertexBuffer* CreateVertexBuffer(uint32_t dataSize, uint32_t strideSize, const void* data) override;
        IRenderTarget* OpenRenderTarget(const RenderTargetDeclaration& declaration, HANDLE handle) override;
        void CopyResource(IRenderTarget* destination, IRenderTarget* source) override;
        // TODO: Context calls - these will need to be moved later.
        void BeginRender() override;
        void EndRender() override;
        void BeginPass(IRenderTarget* renderTarget, const Color& clearColor) override;
        void EndPass() override;
        void SetShader(IShader* shader) override;
        void SetTexture(ITexture* texture) override;
        void SetViewport(const Viewport& viewport) override;
        void SetVertexBuffer(IVertexBuffer* vertexBuffer, uint32_t stride) override;
        void SetIndexBuffer(IIndexBuffer* indexBuffer) override;
        void DrawIndexedPrimitives(uint32_t vertexCount, uint32_t indexCount) override;
        CComPtr<ID3D12Debug3>               m_debug;
        CComPtr<ID3D12Device6>              m_device;
        CComPtr<ID3D12RootSignature>        m_rootSignature;
        CComPtr<ID3D12DescriptorHeap>       m_descriptorHeapGPU;
        CComPtr<ID3D12DescriptorHeap>       m_descriptorHeapGPUSampler;
        CComPtr<ID3D12DescriptorHeap>       m_descriptorHeapCPU;
        CComPtr<ID3D12CommandQueue>         m_commandQueue;
        CComPtr<ID3D12CommandAllocator>     m_commandAllocator;
        CComPtr<ID3D12GraphicsCommandList5> m_frameCommandList;
    };
}