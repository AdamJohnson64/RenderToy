#include "D3D12Utility.h"
#include "ErrorD3D.h"
#include <assert.h>
#include <atlbase.h>

namespace Arcturus
{
    uint32_t D3D12Align(uint32_t size, uint32_t alignSize)
    {
        return size == 0 ? 0 : ((size - 1) / alignSize + 1) * alignSize;
    }

    ID3D12Resource1* D3D12CreateBuffer(ID3D12Device* device, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, uint32_t bufferSize)
    {
        D3D12_HEAP_PROPERTIES descHeapProperties = {};
        descHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC descResource = {};
        descResource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        descResource.Width = bufferSize;
        descResource.Height = 1;
        descResource.DepthOrArraySize = 1;
        descResource.MipLevels = 1;
        descResource.SampleDesc.Count = 1;
        descResource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        descResource.Flags = flags;
        ID3D12Resource1* result = nullptr;
        TRYD3D(device->CreateCommittedResource(&descHeapProperties, D3D12_HEAP_FLAG_NONE, &descResource, state, nullptr, __uuidof(ID3D12Resource1), (void**)&result));
        TRYD3D(result->SetName(L"D3D12CreateBuffer"));
        return result;
    }

    ID3D12Resource1* D3D12CreateBuffer(ID3D12Device* device, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, uint32_t bufferSize, uint32_t dataSize, const void* data, ID3D12CommandQueue* commandQueue, ID3D12CommandAllocator *commandAllocator)
    {
        ID3D12Resource1* result = D3D12CreateBuffer(device, flags, D3D12_RESOURCE_STATE_COPY_DEST, bufferSize);
        CComPtr<ID3D12Resource1> upload;
        D3D12_HEAP_PROPERTIES descHeapProperties = {};
        descHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC descResource = {};
        descResource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        descResource.Width = bufferSize;
        descResource.Height = 1;
        descResource.DepthOrArraySize = 1;
        descResource.MipLevels = 1;
        descResource.SampleDesc.Count = 1;
        descResource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        TRYD3D(device->CreateCommittedResource(&descHeapProperties, D3D12_HEAP_FLAG_NONE, &descResource, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, __uuidof(ID3D12Resource1), (void**)&upload.p));
        TRYD3D(upload->SetName(L"D3D12CreateBuffer_UPLOAD"));
        void *pTLAS = nullptr;
        TRYD3D(upload->Map(0, nullptr, &pTLAS));
        memcpy(pTLAS, data, dataSize);
        upload->Unmap(0, nullptr);
        // Copy this staging buffer to the GPU-only buffer.
        CComPtr<ID3D12GraphicsCommandList5> uploadCommandList;
        TRYD3D(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&uploadCommandList.p));
        uploadCommandList->CopyResource(result, upload);
        {
            D3D12_RESOURCE_BARRIER descBarrier = {};
            descBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            descBarrier.Transition.pResource = result;
            descBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            descBarrier.Transition.StateAfter = state;
            uploadCommandList->ResourceBarrier(1, &descBarrier);
        }
        uploadCommandList->Close();
        commandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&uploadCommandList.p);
        D3D12WaitForGPUIdle(device, commandQueue);
        return result;
    }

    void D3D12WaitForGPUIdle(ID3D12Device* device, ID3D12CommandQueue* queue)
    {
        CComPtr<ID3D12Fence1> fence;
        TRYD3D(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence1), (void**)&fence.p));
        TRYD3D(queue->Signal(fence, 1));
        HANDLE wait = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        assert(wait != nullptr);
        TRYD3D(fence->SetEventOnCompletion(1, wait));
        WaitForSingleObject(wait, INFINITE);
        DeleteObject(wait);
    }
}