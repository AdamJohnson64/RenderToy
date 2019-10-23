#include "IRaytrace_D3D12.h"
#include "MDevice3D.h"
#include "MStub.h"

namespace Arcturus
{
    namespace Managed
    {
        ref class IConstantBuffer_Stub : public CObjectStub<Arcturus::IConstantBuffer>, public IConstantBuffer
        {
        public:
            IConstantBuffer_Stub(Arcturus::IConstantBuffer* object) : CObjectStub(object)
            {
            }
        };

        ref class IConstantBufferView_Stub : public CObjectStub<Arcturus::IConstantBufferView>, public IConstantBufferView
        {
        public:
            IConstantBufferView_Stub(Arcturus::IConstantBufferView* object) : CObjectStub(object)
            {
            }
        };

        ref class IIndexBuffer_Stub : public CObjectStub<Arcturus::IIndexBuffer>, public IIndexBuffer
        {
        public:
            IIndexBuffer_Stub(Arcturus::IIndexBuffer* object) : CObjectStub(object)
            {
            }
        };

        ref class IRenderTarget_Stub : public CObjectStub<Arcturus::IRenderTarget>, public IRenderTarget
        {
        public:
            IRenderTarget_Stub(Arcturus::IRenderTarget* object) : CObjectStub(object)
            {
            }
        };

        ref class IShader_Stub : public CObjectStub<Arcturus::IShader>, public IShader
        {
        public:
            IShader_Stub(Arcturus::IShader* object) : CObjectStub(object)
            {
            }
        };

        ref class IVertexBuffer_Stub : public CObjectStub<Arcturus::IVertexBuffer>, public IVertexBuffer
        {
        public:
            IVertexBuffer_Stub(Arcturus::IVertexBuffer* object) : CObjectStub(object)
            {
            }
        };

        ref class IDevice3D_Stub : public CObjectStub<Arcturus::IDevice3D>, public IDevice3D
        {
        public:
            IDevice3D_Stub(Arcturus::IDevice3D* device) : CObjectStub(device)
            {
            }
            virtual IConstantBuffer^ CreateConstantBuffer(uint32_t dataSize, System::IntPtr data)
            {
                return gcnew IConstantBuffer_Stub(Typed()->CreateConstantBuffer(dataSize, data.ToPointer()));
            }
            virtual IConstantBufferView^ CreateConstantBufferView(IConstantBuffer^ constantBuffer)
            {
                return gcnew IConstantBufferView_Stub(Typed()->CreateConstantBufferView(GetStubTarget<Arcturus::IConstantBuffer>(constantBuffer)));
            }
            virtual IIndexBuffer^ CreateIndexBuffer(uint32_t dataSize, System::IntPtr data)
            {
                return gcnew IIndexBuffer_Stub(Typed()->CreateIndexBuffer(dataSize, data.ToPointer()));
            }
            virtual IRenderTarget^ CreateRenderTarget(RenderTargetDeclaration declaration)
            {
                return gcnew IRenderTarget_Stub(Typed()->CreateRenderTarget(reinterpret_cast<Arcturus::RenderTargetDeclaration&>(declaration)));
            }
            virtual IShader^ CreateShader()
            {
                return gcnew IShader_Stub(Typed()->CreateShader());
            }
            virtual IVertexBuffer^ CreateVertexBuffer(uint32_t dataSize, uint32_t strideSize, System::IntPtr data)
            {
                return gcnew IVertexBuffer_Stub(Typed()->CreateVertexBuffer(dataSize, strideSize, data.ToPointer()));
            }
            virtual IRenderTarget^ OpenRenderTarget(RenderTargetDeclaration declaration, System::IntPtr handle)
            {
                return gcnew IRenderTarget_Stub(Typed()->OpenRenderTarget(reinterpret_cast<Arcturus::RenderTargetDeclaration&>(declaration), handle.ToPointer()));
            }
            virtual void CopyResource(IRenderTarget^ destination, IRenderTarget^ source)
            {
                Typed()->CopyResource(GetStubTarget<Arcturus::IRenderTarget>(destination), GetStubTarget<Arcturus::IRenderTarget>(source));
            }
            virtual void BeginRender()
            {
                Typed()->BeginRender();
            }
            virtual void EndRender()
            {
                Typed()->EndRender();
            }
            virtual void BeginPass(IRenderTarget^ renderTarget, Color color)
            {
                Arcturus::Color colorMarshal;
                colorMarshal.r = color.r;
                colorMarshal.g = color.g;
                colorMarshal.b = color.b;
                colorMarshal.a = color.a;
                Typed()->BeginPass(GetStubTarget<Arcturus::IRenderTarget>(renderTarget), colorMarshal);
            }
            virtual void EndPass()
            {
                Typed()->EndPass();
            }
            virtual void SetShader(IShader^ shader)
            {
                Typed()->SetShader(GetStubTarget<Arcturus::IShader>(shader));
            }
            virtual void SetViewport(Viewport viewport)
            {
                Arcturus::Viewport viewportManaged = {};
                viewportManaged.x = viewport.x;
                viewportManaged.y = viewport.y;
                viewportManaged.width = viewport.width;
                viewportManaged.height = viewport.height;
                viewportManaged.minDepth = viewport.minDepth;
                viewportManaged.maxDepth = viewport.maxDepth;
                Typed()->SetViewport(viewportManaged);
            }
            virtual void SetVertexBuffer(IVertexBuffer^ vertexBuffer, uint32_t stride)
            {
                Typed()->SetVertexBuffer(GetStubTarget<Arcturus::IVertexBuffer>(vertexBuffer), stride);
            }
            virtual void SetIndexBuffer(IIndexBuffer^ indexBuffer)
            {
                Typed()->SetIndexBuffer(GetStubTarget<Arcturus::IIndexBuffer>(indexBuffer));
            }
            virtual void DrawIndexedPrimitives(uint32_t vertexCount, uint32_t indexCount)
            {
                Typed()->DrawIndexedPrimitives(vertexCount, indexCount);
            }
        };

        IDevice3D^ IDevice3D::CreateDevice3D_Direct3D12()
        {
            return gcnew IDevice3D_Stub(Arcturus::CreateDevice3D_Direct3D12());
        }

        IDevice3D^ IDevice3D::CreateDevice3D_Vulkan()
        {
            return gcnew IDevice3D_Stub(Arcturus::CreateDevice3D_Vulkan());
        }

        System::IntPtr IDevice3D::GetID3D12Texture2DHandleNT(System::Object^ object)
        {
            return System::IntPtr(GetStubTarget<Arcturus::IRenderTarget_D3D12>(object)->m_handleNT);
        }

        void IDevice3D::TestRaytracer(IDevice3D^ device, IRenderTarget^ renderTarget, IVertexBuffer^ vertexBuffer, IIndexBuffer^ indexBuffer)
        {
            auto device12 = GetStubTarget<Arcturus::IDevice3D_D3D12>(device);
            auto renderTarget12 = GetStubTarget<Arcturus::IRenderTarget_D3D12>(renderTarget);
            auto vertexBuffer12 = GetStubTarget<Arcturus::IVertexBuffer_D3D12>(vertexBuffer);
            auto indexBuffer12 = GetStubTarget<Arcturus::IIndexBuffer_D3D12>(indexBuffer);
            return Arcturus::TestRaytracer(device12, renderTarget12, vertexBuffer12, indexBuffer12);
        }
    }
}