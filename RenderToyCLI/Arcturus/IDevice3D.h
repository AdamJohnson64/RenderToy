#pragma once

#include "IConstantBuffer.h"
#include "IConstantBufferView.h"
#include "IIndexBuffer.h"
#include "IObject.h"
#include "IRenderTarget.h"
#include "IShader.h"
#include "ITexture.h"
#include "IVertexBuffer.h"
#include "Types3D.h"

#include <Windows.h>

#include <stdint.h>

namespace Arcturus
{
    struct Color
    {
        float r;
        float g;
        float b;
        float a;
    };
    struct Viewport
    {
        float   x;
        float   y;
        float   width;
        float   height;
        float   minDepth;
        float   maxDepth;
    };
    class IDevice3D : public IObject
    {
    public:
        virtual IConstantBuffer* CreateConstantBuffer(uint32_t dataSize, const void* data) = 0;
        virtual IConstantBufferView* CreateConstantBufferView(IConstantBuffer* constantBuffer) = 0;
        virtual IIndexBuffer* CreateIndexBuffer(uint32_t dataSize, const void* data) = 0;
        virtual IRenderTarget* CreateRenderTarget(const RenderTargetDeclaration& declaration) = 0;
        virtual IShader* CreateShader() = 0;
        virtual ITexture* CreateTexture2D(uint32_t width, uint32_t height, const void* data) = 0;
        virtual IVertexBuffer* CreateVertexBuffer(uint32_t dataSize, uint32_t strideSize, const void* data) = 0;
        virtual IRenderTarget* OpenRenderTarget(const RenderTargetDeclaration& declaration, HANDLE handle) = 0;
        virtual void CopyResource(IRenderTarget* destination, IRenderTarget* source) = 0;
        // TODO: Context calls - these will need to be moved later.
        virtual void BeginRender() = 0;
        virtual void EndRender() = 0;
        virtual void BeginPass(IRenderTarget* renderTarget, const Color& clearColor) = 0;
        virtual void EndPass() = 0;
        virtual void SetShader(IShader* shader) = 0;
        virtual void SetTexture(ITexture* texture) = 0;
        virtual void SetViewport(const Viewport& viewport) = 0;
        virtual void SetVertexBuffer(IVertexBuffer* vertexBuffer, uint32_t stride) = 0;
        virtual void SetIndexBuffer(IIndexBuffer* indexBuffer) = 0;
        virtual void DrawIndexedPrimitives(uint32_t vertexCount, uint32_t indexCount) = 0;
    };
    IDevice3D* CreateDevice3D_Direct3D12();
    IDevice3D* CreateDevice3D_Vulkan();
}