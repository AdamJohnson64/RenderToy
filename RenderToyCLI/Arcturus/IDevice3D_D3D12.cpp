#include "D3D12Utility.h"
#include "ErrorD3D.h"
#include "IConstantBuffer_D3D12.h"
#include "IConstantBufferView_D3D12.h"
#include "IIndexBuffer_D3D12.h"
#include "IRenderTarget_D3D12.h"
#include "IShader_D3D12.h"
#include "ITexture_D3D12.h"
#include "IVertexBuffer_D3D12.h"
#include <array>
#include <assert.h>

constexpr int GPU_DESCRIPTOR_COUNT = 65536;
constexpr int GPU_DESCRIPTOR_SAMPLER_COUNT = 64;
constexpr int GPU_DESCRIPTOR_CBV_COMMON = 0;
constexpr int GPU_DESCRIPTOR_SRV_COMMON = 1;
constexpr int GPU_DESCRIPTOR_SMP_COMMON = 2;

namespace Arcturus
{
    IDevice3D_D3D12::IDevice3D_D3D12()
    {
        // [DXR] Enable DXIL shader compilation for shader model 6.x
        {
            UUID Features[1] = {
                D3D12ExperimentalShaderModels,
            };
            D3D12EnableExperimentalFeatures(1, Features, nullptr, nullptr);
        }
        // Create the D3D12 device.
        TRYD3D(D3D12GetDebugInterface(__uuidof(ID3D12Debug3), (void**)&m_debug.p));
        m_debug->EnableDebugLayer();
        TRYD3D(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, __uuidof(ID3D12Device6), (void**)&m_device.p));
        {
            std::array<D3D12_DESCRIPTOR_RANGE, 3> descDescriptorRange = {};
            descDescriptorRange[GPU_DESCRIPTOR_CBV_COMMON].BaseShaderRegister = 0;
	        descDescriptorRange[GPU_DESCRIPTOR_CBV_COMMON].NumDescriptors = 1;
	        descDescriptorRange[GPU_DESCRIPTOR_CBV_COMMON].RegisterSpace = 0;
	        descDescriptorRange[GPU_DESCRIPTOR_CBV_COMMON].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
	        descDescriptorRange[GPU_DESCRIPTOR_CBV_COMMON].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            descDescriptorRange[GPU_DESCRIPTOR_SRV_COMMON].BaseShaderRegister = 0;
	        descDescriptorRange[GPU_DESCRIPTOR_SRV_COMMON].NumDescriptors = 1;
	        descDescriptorRange[GPU_DESCRIPTOR_SRV_COMMON].RegisterSpace = 0;
	        descDescriptorRange[GPU_DESCRIPTOR_SRV_COMMON].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
	        descDescriptorRange[GPU_DESCRIPTOR_SRV_COMMON].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            descDescriptorRange[GPU_DESCRIPTOR_SMP_COMMON].BaseShaderRegister = 0;
            descDescriptorRange[GPU_DESCRIPTOR_SMP_COMMON].NumDescriptors = 1;
            descDescriptorRange[GPU_DESCRIPTOR_SMP_COMMON].RegisterSpace = 0;
            descDescriptorRange[GPU_DESCRIPTOR_SMP_COMMON].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            descDescriptorRange[GPU_DESCRIPTOR_SMP_COMMON].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            std::array<D3D12_ROOT_PARAMETER, 3> descRootParameter = {};
	        descRootParameter[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	        descRootParameter[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	        descRootParameter[0].DescriptorTable.NumDescriptorRanges = 1;
	        descRootParameter[0].DescriptorTable.pDescriptorRanges = &descDescriptorRange[0];

	        descRootParameter[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	        descRootParameter[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	        descRootParameter[1].DescriptorTable.NumDescriptorRanges = 1;
	        descRootParameter[1].DescriptorTable.pDescriptorRanges = &descDescriptorRange[1];

	        descRootParameter[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	        descRootParameter[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	        descRootParameter[2].DescriptorTable.NumDescriptorRanges = 1;
	        descRootParameter[2].DescriptorTable.pDescriptorRanges = &descDescriptorRange[2];

            D3D12_ROOT_SIGNATURE_DESC descSignature = {};
	        descSignature.NumParameters = descRootParameter.size();
	        descSignature.pParameters = descRootParameter.data();
	        descSignature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

            CComPtr<ID3DBlob> m_blob, m_blobError;
            TRYD3D(D3D12SerializeRootSignature(&descSignature, D3D_ROOT_SIGNATURE_VERSION_1_0, &m_blob.p, &m_blobError.p));
            TRYD3D(m_device->CreateRootSignature(0, m_blob->GetBufferPointer(), m_blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&m_rootSignature.p));
        }
        {
            D3D12_DESCRIPTOR_HEAP_DESC descDescriptorHeap = {};
            descDescriptorHeap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            descDescriptorHeap.NumDescriptors = GPU_DESCRIPTOR_COUNT;
            descDescriptorHeap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            TRYD3D(m_device->CreateDescriptorHeap(&descDescriptorHeap, __uuidof(ID3D12DescriptorHeap), (void**)&m_descriptorHeapGPU.p));
        }
        {
            D3D12_DESCRIPTOR_HEAP_DESC descDescriptorHeap = {};
            descDescriptorHeap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            descDescriptorHeap.NumDescriptors = GPU_DESCRIPTOR_SAMPLER_COUNT;
            descDescriptorHeap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            TRYD3D(m_device->CreateDescriptorHeap(&descDescriptorHeap, __uuidof(ID3D12DescriptorHeap), (void**)&m_descriptorHeapGPUSampler.p));
        }
        {
            D3D12_DESCRIPTOR_HEAP_DESC descDescriptorHeap = {};
            descDescriptorHeap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
            descDescriptorHeap.NumDescriptors = 64;
            TRYD3D(m_device->CreateDescriptorHeap(&descDescriptorHeap, __uuidof(ID3D12DescriptorHeap), (void**)&m_descriptorHeapCPU.p));
        }
        {
            D3D12_COMMAND_QUEUE_DESC descCommandQueue = {};
            TRYD3D(m_device->CreateCommandQueue(&descCommandQueue, __uuidof(ID3D12CommandQueue), (void**)&m_commandQueue.p));
        }
        TRYD3D(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&m_commandAllocator.p));
        // Create a sampler that everything can use (for now).
        {
	        UINT descriptorElementSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_CPU_DESCRIPTOR_HANDLE handleCPU = m_descriptorHeapGPUSampler->GetCPUDescriptorHandleForHeapStart();
            handleCPU.ptr = handleCPU.ptr + descriptorElementSize * GPU_DESCRIPTOR_SMP_COMMON;
            D3D12_SAMPLER_DESC descSampler = {};
            descSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            descSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            descSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            descSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            descSampler.MaxLOD = 1;
            m_device->CreateSampler(&descSampler, handleCPU);
        }
    }

    IConstantBuffer* IDevice3D_D3D12::CreateConstantBuffer(uint32_t dataSize, const void* data)
    {
        return new IConstantBuffer_D3D12(this, dataSize, data);
    }

    IConstantBufferView* IDevice3D_D3D12::CreateConstantBufferView(IConstantBuffer* constantBuffer)
    {
        return new IConstantBufferView_D3D12(this, dynamic_cast<IConstantBuffer_D3D12*>(constantBuffer));
    }

    IIndexBuffer* IDevice3D_D3D12::CreateIndexBuffer(uint32_t dataSize, const void* data)
    {
        return new IIndexBuffer_D3D12(this, dataSize, data);
    }

    IRenderTarget* IDevice3D_D3D12::CreateRenderTarget(const RenderTargetDeclaration& declaration)
    {
        return new IRenderTarget_D3D12(this, declaration);
    }

    IShader* IDevice3D_D3D12::CreateShader()
    {
        return new IShader_D3D12(this);
    }

    ITexture* IDevice3D_D3D12::CreateTexture2D(uint32_t width, uint32_t height, const void* data)
    {
        return new ITexture_D3D12(this, width, height, data);
    }

    IVertexBuffer* IDevice3D_D3D12::CreateVertexBuffer(uint32_t dataSize, uint32_t strideSize, const void* data)
    {
        return new IVertexBuffer_D3D12(this, dataSize, strideSize, data);
    }

    IRenderTarget* IDevice3D_D3D12::OpenRenderTarget(const RenderTargetDeclaration& declaration, HANDLE handle)
    {
        return new IRenderTarget_D3D12(this, declaration, handle);
    }

    void IDevice3D_D3D12::CopyResource(IRenderTarget* destination, IRenderTarget* source)
    {
        IRenderTarget_D3D12* destination12 = dynamic_cast<IRenderTarget_D3D12*>(destination);
        IRenderTarget_D3D12* source12 = dynamic_cast<IRenderTarget_D3D12*>(source);
        CComPtr<ID3D12GraphicsCommandList> commandList;
        TRYD3D(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&commandList.p));
        commandList->CopyResource(destination12->m_resource, source12->m_resource);
        TRYD3D(commandList->Close());
        m_commandQueue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&commandList.p));
        D3D12WaitForGPUIdle(m_device, m_commandQueue);
    }

    // TODO: Context calls - these will need to be moved later.
    void IDevice3D_D3D12::BeginRender()
    {
        assert(m_frameCommandList.p == nullptr);
        TRYD3D(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&m_frameCommandList.p));
        m_frameCommandList->SetGraphicsRootSignature(m_rootSignature);
        {
            ID3D12DescriptorHeap* descHeaps[2] = { m_descriptorHeapGPU, m_descriptorHeapGPUSampler };
            m_frameCommandList->SetDescriptorHeaps(_countof(descHeaps), descHeaps);
        }
        m_frameCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        {
	        UINT descriptorElementSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_GPU_DESCRIPTOR_HANDLE handleGPU = m_descriptorHeapGPU->GetGPUDescriptorHandleForHeapStart();
            handleGPU.ptr = handleGPU.ptr + descriptorElementSize * GPU_DESCRIPTOR_CBV_COMMON;
            m_frameCommandList->SetGraphicsRootDescriptorTable(GPU_DESCRIPTOR_CBV_COMMON, handleGPU);
        }
        {
	        UINT descriptorElementSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            D3D12_GPU_DESCRIPTOR_HANDLE handleGPU = m_descriptorHeapGPUSampler->GetGPUDescriptorHandleForHeapStart();
            handleGPU.ptr = handleGPU.ptr + descriptorElementSize * GPU_DESCRIPTOR_SMP_COMMON;
            m_frameCommandList->SetGraphicsRootDescriptorTable(GPU_DESCRIPTOR_SMP_COMMON, handleGPU);
        }
    }

    void IDevice3D_D3D12::EndRender()
    {
        assert(m_frameCommandList != nullptr);
        m_frameCommandList->Close();
        m_commandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&m_frameCommandList.p);
        D3D12WaitForGPUIdle(m_device, m_commandQueue);
        m_commandAllocator->Reset();
        m_frameCommandList.Release();
    }

    void IDevice3D_D3D12::BeginPass(IRenderTarget* renderTarget, const Color& clearColor)
    {
        IRenderTarget_D3D12* renderTarget12 = dynamic_cast<IRenderTarget_D3D12*>(renderTarget);
        D3D12_CPU_DESCRIPTOR_HANDLE m_descDescriptorHandle = m_descriptorHeapCPU->GetCPUDescriptorHandleForHeapStart();
        {
            D3D12_RENDER_TARGET_VIEW_DESC descRenderTargetView = {};
            descRenderTargetView.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            descRenderTargetView.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            m_device->CreateRenderTargetView(renderTarget12->m_resource, &descRenderTargetView, m_descDescriptorHandle);
        }
        D3D12_RENDER_PASS_RENDER_TARGET_DESC descRenderTarget = {};
        descRenderTarget.cpuDescriptor = m_descriptorHeapCPU->GetCPUDescriptorHandleForHeapStart();
        descRenderTarget.BeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
        descRenderTarget.BeginningAccess.Clear.ClearValue.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        (Color&)descRenderTarget.BeginningAccess.Clear.ClearValue.Color = clearColor;
        descRenderTarget.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
        m_frameCommandList->BeginRenderPass(1, &descRenderTarget, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
    }
    
    void IDevice3D_D3D12::EndPass()
    {
        m_frameCommandList->EndRenderPass();
    }

    void IDevice3D_D3D12::SetShader(IShader* shader)
    {
        IShader_D3D12* shader12 = dynamic_cast<IShader_D3D12*>(shader);
        m_frameCommandList->SetPipelineState(shader12->m_pipelineState);
    }

    void IDevice3D_D3D12::SetTexture(ITexture* texture)
    {
        ITexture_D3D12* texture12 = dynamic_cast<ITexture_D3D12*>(texture);
	    UINT descriptorElementSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE handleCPU = m_descriptorHeapGPU->GetCPUDescriptorHandleForHeapStart();
        handleCPU.ptr = handleCPU.ptr + descriptorElementSize * GPU_DESCRIPTOR_SRV_COMMON;
        m_device->CreateShaderResourceView(texture12->m_resource, &texture12->m_view, handleCPU);
        D3D12_GPU_DESCRIPTOR_HANDLE handleGPU = m_descriptorHeapGPU->GetGPUDescriptorHandleForHeapStart();
        handleGPU.ptr = handleGPU.ptr + descriptorElementSize * GPU_DESCRIPTOR_SRV_COMMON;
        m_frameCommandList->SetGraphicsRootDescriptorTable(1, handleGPU);
    }

    void IDevice3D_D3D12::SetViewport(const Viewport& viewport)
    {
        m_frameCommandList->RSSetViewports(1, reinterpret_cast<const D3D12_VIEWPORT*>(&viewport));
        D3D12_RECT descScissor = {};
        descScissor.left = viewport.x;
        descScissor.top = viewport.y;
        descScissor.right = viewport.x + viewport.width;
        descScissor.bottom = viewport.y + viewport.height;
        m_frameCommandList->RSSetScissorRects(1, &descScissor);
    }

    void IDevice3D_D3D12::SetVertexBuffer(IVertexBuffer* vertexBuffer, uint32_t stride)
    {
        m_frameCommandList->IASetVertexBuffers(0, 1, &dynamic_cast<IVertexBuffer_D3D12*>(vertexBuffer)->m_descVertexBufferView);
    }

    void IDevice3D_D3D12::SetIndexBuffer(IIndexBuffer* indexBuffer)
    {
        m_frameCommandList->IASetIndexBuffer(&dynamic_cast<IIndexBuffer_D3D12*>(indexBuffer)->m_descIndexBufferView);
    }

    void IDevice3D_D3D12::DrawIndexedPrimitives(uint32_t vertexCount, uint32_t indexCount)
    {
        m_frameCommandList->DrawIndexedInstanced(indexCount, 1, 0, 0, 0);
    }

    IDevice3D* CreateDevice3D_Direct3D12()
    {
        return new IDevice3D_D3D12();
    }
}