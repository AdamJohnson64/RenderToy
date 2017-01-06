#include <d3d9.h>

namespace RenderToy
{
	public ref class D3D9GlobalServices {
	private:
		D3D9GlobalServices() {
			hHostWindow = CreateWindow("STATIC", "D3D9HostWindow", WS_OVERLAPPEDWINDOW, 0, 0, 16, 16, nullptr, nullptr, nullptr, nullptr);
			if (hHostWindow == nullptr) {
				throw gcnew System::Exception("CreateWindow() failed.");
			}
			pD3D = Direct3DCreate9(D3D_SDK_VERSION);
			if (pD3D == nullptr) {
				throw gcnew System::Exception("Direct3DCreate9() failed.");
			}
		}
		!D3D9GlobalServices() {
			Destroy();
		}
		~D3D9GlobalServices() {
			Destroy();
		}
		void Destroy() {
			if (pD3D != nullptr) {
				pD3D->Release();
				pD3D = nullptr;
			}
			if (hHostWindow != nullptr) {
				DestroyWindow(hHostWindow);
				hHostWindow = nullptr;
			}
		}
	public:
		HWND hHostWindow = nullptr;
		IDirect3D9* pD3D = nullptr;
		static D3D9GlobalServices^ Instance = gcnew D3D9GlobalServices();
	};
	public ref class D3D9Surface {
	public:
		D3D9Surface() {
			D3DPRESENT_PARAMETERS d3dpp = { 0 };
			d3dpp.BackBufferWidth = 256;
			d3dpp.BackBufferHeight = 256;
			d3dpp.BackBufferFormat = D3DFMT_A8R8G8B8;
			d3dpp.BackBufferCount = 1;
			d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
			d3dpp.Windowed = TRUE;
			d3dpp.EnableAutoDepthStencil = TRUE;
			d3dpp.AutoDepthStencilFormat = D3DFMT_D24S8;
			IDirect3DDevice9* pDeviceTmp = nullptr;
			if (D3D9GlobalServices::Instance->pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, D3D9GlobalServices::Instance->hHostWindow, D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, &pDeviceTmp) != D3D_OK) {
				throw gcnew System::Exception("IDirect3D9::CreateDevice() failed.");
			}
			pDevice = pDeviceTmp;
			IDirect3DSurface9 *pSurfaceTmp = nullptr;
			if (pDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &pSurfaceTmp) != D3D_OK) {
				throw gcnew System::Exception("IDirect3DDevice9::GetBackBuffer() failed.");
			}
			pSurface = pSurfaceTmp;
			if (pDevice->Clear(0, nullptr, D3DCLEAR_TARGET, 0xffff80ff, 0.0, 0) != D3D_OK) {
				throw gcnew System::Exception("IDirect3DDevice9::Clear() failed.");
			}
			D3DRECT rect = { 0, 0, 16, 16 };
			if (pDevice->Clear(1, &rect, D3DCLEAR_TARGET, 0xffff0000, 0.0, 0) != D3D_OK) {
				throw gcnew System::Exception("IDirect3DDevice9::Clear() failed.");
			}
		}
		!D3D9Surface() {
			Destroy();
		}
		~D3D9Surface() {
			Destroy();
		}
		void Destroy() {
			if (pSurface != nullptr) {
				pSurface->Release();
				pSurface = nullptr;
			}
			if (pDevice != nullptr) {
				pDevice->Release();
				pDevice = nullptr;
			}
		}
		property System::IntPtr SurfacePtr {
			System::IntPtr get() { return System::IntPtr(pSurface); }
		}
		IDirect3DDevice9* pDevice = nullptr;
		IDirect3DSurface9* pSurface = nullptr;
	};
}