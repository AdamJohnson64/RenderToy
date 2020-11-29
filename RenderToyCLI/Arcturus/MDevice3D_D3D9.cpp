#include "ErrorD3D.h"
#include "MTypes3D.h"
#include "Vector.h"
#include <atlbase.h>
#include <d3d9.h>
#include <stdint.h>

namespace Arcturus
{
    class D3D9Globals
    {
    public:
        D3D9Globals()
        {
            // Create a fake backing window.
            m_Window = CreateWindow(L"BUTTON", L"BUTTON", WS_OVERLAPPEDWINDOW, 0, 0, 1, 1, nullptr, nullptr, nullptr, nullptr);
            if (m_Window == nullptr) throw std::exception("Failed to create window.");
            // Create the Direct3D interface.
            TRYD3D(Direct3DCreate9Ex(DIRECT3D_VERSION, &m_Direct3D.p));
        }
        HWND m_Window;
        CComPtr<IDirect3D9Ex> m_Direct3D;
    } g_Direct3D9Globals;

    namespace Managed
    {
        ref class IRenderTarget_D3D9;

        public ref class IDevice3D_D3D9
        {
        public:
            IDevice3D_D3D9();
            ~IDevice3D_D3D9();
            !IDevice3D_D3D9();
            IRenderTarget_D3D9^ CreateRenderTarget(RenderTargetDeclaration declaration);
            IDirect3DDevice9Ex* m_device;
        protected:
            void Destroy();
        };

        public ref class IRenderTarget_D3D9
        {
        public:
            IRenderTarget_D3D9(IDevice3D_D3D9^ owner, RenderTargetDeclaration declaration);
            ~IRenderTarget_D3D9();
            !IRenderTarget_D3D9();
            void Fill(System::IntPtr data);
            System::IntPtr GetIDirect3DSurface9Handle();
            System::IntPtr GetIDirect3DSurface9Pointer();
            IDirect3DSurface9* m_surface;
            HANDLE m_handle;
        private:
            void Destroy();
            IDevice3D_D3D9^ m_owner;
        };

        IRenderTarget_D3D9::IRenderTarget_D3D9(IDevice3D_D3D9^ owner, RenderTargetDeclaration declaration) : m_owner(owner)
        {
            IDirect3DSurface9* surface = nullptr;
            HANDLE handle = nullptr;
            TRYD3D(m_owner->m_device->CreateRenderTarget(declaration.width, declaration.height, D3DFMT_A8R8G8B8, D3DMULTISAMPLE_NONE, 0, TRUE, &surface, &handle));
            m_surface = surface;
            m_handle = handle;
        }
            
        IRenderTarget_D3D9::~IRenderTarget_D3D9()
        {
            Destroy();
        }

        IRenderTarget_D3D9::!IRenderTarget_D3D9()
        {
            Destroy();
        }

        void IRenderTarget_D3D9::Fill(System::IntPtr data)
        {
            D3DLOCKED_RECT rect = {};
            TRYD3D(m_surface->LockRect(&rect, nullptr, 0));
            for (int y = 0; y < 256; ++y)
            {
                const void* pRasterIn = (char*)data.ToPointer() + sizeof(uint32_t) * 256 * y;
                void* pRasterOut = (char*)rect.pBits + rect.Pitch * y;
                memcpy(pRasterOut, pRasterIn, sizeof(uint32_t) * 256);
            }
            TRYD3D(m_surface->UnlockRect());
        }

        System::IntPtr IRenderTarget_D3D9::GetIDirect3DSurface9Handle()
        {
            return System::IntPtr(m_handle);
        }

        System::IntPtr IRenderTarget_D3D9::GetIDirect3DSurface9Pointer()
        {
            return System::IntPtr(m_surface);
        }

        void IRenderTarget_D3D9::Destroy()
        {
            if (m_surface != nullptr)
            {
                m_surface->Release();
                m_surface = nullptr;
            }
        }

        IDevice3D_D3D9::IDevice3D_D3D9()
        {
            D3DPRESENT_PARAMETERS desc = {};
            desc.BackBufferWidth = 1;
            desc.BackBufferHeight = 1;
            desc.BackBufferFormat = D3DFMT_UNKNOWN;
            desc.SwapEffect = D3DSWAPEFFECT_DISCARD;
            desc.hDeviceWindow = g_Direct3D9Globals.m_Window;
            desc.Windowed = TRUE;
            IDirect3DDevice9Ex* device = nullptr;
            TRYD3D(g_Direct3D9Globals.m_Direct3D->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, g_Direct3D9Globals.m_Window, D3DCREATE_MULTITHREADED | D3DCREATE_FPU_PRESERVE | D3DCREATE_HARDWARE_VERTEXPROCESSING, &desc, nullptr, &device));
            m_device = device;
        }

        IDevice3D_D3D9::~IDevice3D_D3D9()
        {
            Destroy();
        }

        IDevice3D_D3D9::!IDevice3D_D3D9()
        {
            Destroy();
        }

        IRenderTarget_D3D9^ IDevice3D_D3D9::CreateRenderTarget(RenderTargetDeclaration declaration)
        {
            return gcnew IRenderTarget_D3D9(this, declaration);
        }

        void IDevice3D_D3D9::Destroy()
        {
            if (m_device != nullptr)
            {
                m_device->Release();
                m_device = nullptr;
            }
        }
    }
}