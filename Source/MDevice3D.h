#pragma once

#include "AutoRelease.h"
#include "IDevice3D.h"
#include "MTypes3D.h"

#include <stdint.h>

namespace Arcturus
{
    namespace Managed
    {
        public value struct Color
        {
            float r;
            float g;
            float b;
            float a;
        };

        public value struct Viewport
        {
            float x;
            float y;
            float width;
            float height;
            float minDepth;
            float maxDepth;
        };

        public interface class IConstantBuffer
        {
        };

        public interface class IConstantBufferView
        {
        };

        public interface class IIndexBuffer
        {
        };

        public interface class IRenderTarget
        {
        };

        public interface class IShader
        {
        };

        public interface class ITexture
        {
        };

        public interface class IVertexBuffer
        {
        };

        public interface class IDevice3D
        {
        public:
            static IDevice3D^ CreateDevice3D_Direct3D12();
            static IDevice3D^ CreateDevice3D_Vulkan();
            static void TestRaytracer(IDevice3D^ device, IRenderTarget^ renderTarget, IVertexBuffer^ vertexBuffer, IIndexBuffer^ indexBuffer);
            static System::IntPtr GetID3D12Texture2DHandleNT(System::Object^ object);
            virtual IConstantBuffer^ CreateConstantBuffer(uint32_t dataSize, System::IntPtr data) = 0;
            virtual IConstantBufferView^ CreateConstantBufferView(IConstantBuffer^ constantBuffer) = 0;
            virtual IIndexBuffer^ CreateIndexBuffer(uint32_t dataSize, System::IntPtr data) = 0;
            virtual IRenderTarget^ CreateRenderTarget(RenderTargetDeclaration declaration) = 0;
            virtual IShader^ CreateShader() = 0;
            virtual ITexture^ CreateTexture2D(uint32_t width, uint32_t height, System::IntPtr data) = 0;
            virtual IVertexBuffer^ CreateVertexBuffer(uint32_t dataSize, uint32_t strideSize, System::IntPtr data) = 0;
            virtual IRenderTarget^ OpenRenderTarget(RenderTargetDeclaration declaration, System::IntPtr handle) = 0;
            virtual void CopyResource(IRenderTarget^ destination, IRenderTarget^ source) = 0;
            virtual void BeginRender() = 0;
            virtual void EndRender() = 0;
            virtual void BeginPass(IRenderTarget^ renderTarget, Color clearColor) = 0;
            virtual void EndPass() = 0;
            virtual void SetShader(IShader^ shader) = 0;
            virtual void SetTexture(ITexture^ texture) = 0;
            virtual void SetViewport(Viewport viewport) = 0;
            virtual void SetVertexBuffer(IVertexBuffer^ vertexBuffer, uint32_t stride) = 0;
            virtual void SetIndexBuffer(IIndexBuffer^ indexBuffer) = 0;
            virtual void DrawIndexedPrimitives(uint32_t vertexCount, uint32_t indexCount) = 0;
        };
    }
}