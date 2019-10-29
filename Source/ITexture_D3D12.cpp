#include "D3D12Utility.h"
#include "ErrorD3D.h"
#include "IDevice3D_D3D12.h"
#include "ITexture_D3D12.h"

#include <exception>

namespace Arcturus
{
    ITexture_D3D12::ITexture_D3D12(IDevice3D_D3D12* owner, uint32_t width, uint32_t height, const void* data) : m_owner(owner)
    {
        // Create the final output texture in an uninitialized state.
        {
            D3D12_HEAP_PROPERTIES descHeap = {};
            descHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC descResource = {};
            descResource.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            descResource.Width = width;
            descResource.Height = height;
            descResource.DepthOrArraySize = 1;
            descResource.MipLevels = 1;
            descResource.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            descResource.SampleDesc.Count = 1;
            descResource.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            TRYD3D(owner->m_device->CreateCommittedResource1(&descHeap, D3D12_HEAP_FLAG_NONE, &descResource, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, _uuidof(ID3D12Resource1), (void**)&m_resource));
            m_view = D3D12_SHADER_RESOURCE_VIEW_DESC {};
            m_view.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            m_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            m_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            m_view.Texture2D.MipLevels = 1;
        }
        // Create an upload buffer and blit the image data over to the GPU resource.
        {
            // Create a buffer in CPU visible memory.
            AutoRelease<ID3D12Resource1> resourceUpload;
            {
                D3D12_HEAP_PROPERTIES descHeap = {};
                descHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
                D3D12_RESOURCE_DESC descResource = {};
                descResource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                descResource.Width = sizeof(uint32_t) * width * height;
                descResource.Height = 1;
                descResource.DepthOrArraySize = 1;
                descResource.MipLevels = 1;
                descResource.Format = DXGI_FORMAT_UNKNOWN;
                descResource.SampleDesc.Count = 1;
                descResource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                TRYD3D(owner->m_device->CreateCommittedResource1(&descHeap, D3D12_HEAP_FLAG_NONE, &descResource, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, nullptr, _uuidof(ID3D12Resource1), (void**)&resourceUpload));
            }
            // Map and copy up the data to the CPU buffer.
            {
                void *pData = nullptr;
                TRYD3D(resourceUpload->Map(0, nullptr, &pData));
                memcpy(pData, data, sizeof(uint32_t) * width * height);
                resourceUpload->Unmap(0, nullptr);
            }
            // Copy this staging buffer to the GPU-only buffer.
            AutoRelease<ID3D12GraphicsCommandList5> uploadCommandList;
            TRYD3D(m_owner->m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_owner->m_commandAllocator, nullptr, __uuidof(ID3D12GraphicsCommandList5), (void**)&uploadCommandList));
            {
                D3D12_TEXTURE_COPY_LOCATION descDst = {};
                descDst.pResource = m_resource;
                descDst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                D3D12_TEXTURE_COPY_LOCATION descSrc = {};
                descSrc.pResource = resourceUpload;
                descSrc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                descSrc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                descSrc.PlacedFootprint.Footprint.Width = width;
                descSrc.PlacedFootprint.Footprint.Height = height;
                descSrc.PlacedFootprint.Footprint.Depth = 1;
                descSrc.PlacedFootprint.Footprint.RowPitch = sizeof(uint32_t) * width;
                uploadCommandList->CopyTextureRegion(&descDst, 0, 0, 0, &descSrc, nullptr);
            }
            uploadCommandList->Close();
            m_owner->m_commandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&uploadCommandList);
            D3D12WaitForGPUIdle(m_owner->m_device, m_owner->m_commandQueue);
        }
    }
}