#include "D3D12Utility.h"
#include "ErrorD3D.h"
#include "IConstantBuffer_D3D12.h"
#include "IConstantBufferView_D3D12.h"
#include "IIndexBuffer_D3D12.h"
#include "IRenderTarget_D3D12.h"
#include "IShader_D3D12.h"
#include "IVertexBuffer_D3D12.h"

#include <exception>

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
        TRYD3D(D3D12GetDebugInterface(__uuidof(ID3D12Debug3), (void**)&m_debug));
        m_debug->EnableDebugLayer();
        TRYD3D(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, __uuidof(ID3D12Device6), (void**)&m_device));
        {
            uint32_t setupRange = 0;
            uint32_t setupOffset = 0;

	        D3D12_DESCRIPTOR_RANGE descDescriptorRange[32];
            
            descDescriptorRange[setupRange].BaseShaderRegister = 0;
	        descDescriptorRange[setupRange].NumDescriptors = 1;
	        descDescriptorRange[setupRange].RegisterSpace = 0;
	        descDescriptorRange[setupRange].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
	        descDescriptorRange[setupRange].OffsetInDescriptorsFromTableStart = setupOffset;
            setupOffset += descDescriptorRange[setupRange].NumDescriptors;
            ++setupRange;

            D3D12_ROOT_PARAMETER descRootParameter = {};
	        descRootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	        descRootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	        descRootParameter.DescriptorTable.NumDescriptorRanges = setupRange;
	        descRootParameter.DescriptorTable.pDescriptorRanges = descDescriptorRange;

            D3D12_ROOT_SIGNATURE_DESC descSignature = {};
	        descSignature.NumParameters = 1;
	        descSignature.pParameters = &descRootParameter;
	        descSignature.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

            AutoRelease<ID3DBlob> m_blob;
            AutoRelease<ID3DBlob> m_blobError;
            TRYD3D(D3D12SerializeRootSignature(&descSignature, D3D_ROOT_SIGNATURE_VERSION_1_0, &m_blob, &m_blobError));
            TRYD3D(m_device->CreateRootSignature(0, m_blob->GetBufferPointer(), m_blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&m_rootSignature));
        }
        {
            D3D12_DESCRIPTOR_HEAP_DESC descDescriptorHeap = {};
            descDescriptorHeap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            descDescriptorHeap.NumDescriptors = 1;
            descDescriptorHeap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            TRYD3D(m_device->CreateDescriptorHeap(&descDescriptorHeap, __uuidof(ID3D12DescriptorHeap), (void**)&m_descriptorHeapGPU));
        }
        {
            D3D12_DESCRIPTOR_HEAP_DESC descDescriptorHeap = {};
            descDescriptorHeap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
            descDescriptorHeap.NumDescriptors = 64;
            TRYD3D(m_device->CreateDescriptorHeap(&descDescriptorHeap, __uuidof(ID3D12DescriptorHeap), (void**)&m_descriptorHeapCPU));
        }
        {
            D3D12_COMMAND_QUEUE_DESC descCommandQueue = {};
            TRYD3D(m_device->CreateCommandQueue(&descCommandQueue, __uuidof(ID3D12CommandQueue), (void**)&m_commandQueue));
        }
        TRYD3D(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&m_commandAllocator));
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
        AutoRelease<ID3D12GraphicsCommandList> commandList;
        TRYD3D(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&commandList));
        commandList->CopyResource(destination12->m_resource, source12->m_resource);
        TRYD3D(commandList->Close());
        m_commandQueue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&commandList));
        D3D12WaitForGPUIdle(m_device, m_commandQueue);
    }

    // TODO: Context calls - these will need to be moved later.
    void IDevice3D_D3D12::BeginRender()
    {
        assert(m_frameCommandList.p == nullptr);
        TRYD3D(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&m_frameCommandList));
    }

    void IDevice3D_D3D12::EndRender()
    {
        assert(m_frameCommandList != nullptr);
        m_frameCommandList->Close();
        m_commandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&m_frameCommandList);
        D3D12WaitForGPUIdle(m_device, m_commandQueue);
        m_commandAllocator->Reset();
        m_frameCommandList.Destroy();
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
        m_frameCommandList->SetGraphicsRootSignature(m_rootSignature);
        m_frameCommandList->SetDescriptorHeaps(1, &m_descriptorHeapGPU);
        m_frameCommandList->SetGraphicsRootDescriptorTable(0, m_descriptorHeapGPU->GetGPUDescriptorHandleForHeapStart());
        m_frameCommandList->SetPipelineState(shader12->m_pipelineState);
        m_frameCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
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
}