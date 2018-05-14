////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// A simple Fixed Function Direct3D 9 interface presented to the Common Language
// Runtime for consumption by .NET languages.
//
// This interface is highly-simplistic and does not attempt to model persistent
// GPU resources or framebuffers. Loss of device is trivially handled by never
// attempting to reuse devices beyond a single frame.
////////////////////////////////////////////////////////////////////////////////

#include <d3d9.h>

#define TRY_D3D(D3D9FUNC) if ((D3D9FUNC) != D3D_OK) throw gcnew System::Exception(#D3D9FUNC)

struct FlexibleVertex_XYZW_DIFFUSE {
	float x, y, z, w;
	unsigned int color;
};

namespace RenderToy
{
	ref class D3D9GlobalServices {
	private:
		#pragma region - Section : Construction -
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
		#pragma endregion
	public:
		HWND hHostWindow = nullptr;
		IDirect3D9* pD3D = nullptr;
		static D3D9GlobalServices^ Instance = gcnew D3D9GlobalServices();
	};
	public enum class D3DPrimitiveType
	{
		D3DPT_TRIANGLELIST = D3DPT_TRIANGLELIST,
	};
	public ref class D3D9Surface {
	public:
		#pragma region - Section : Construction -
		D3D9Surface(int render_width, int render_height) {
			D3DPRESENT_PARAMETERS d3dpp = { 0 };
			d3dpp.BackBufferWidth = render_width;
			d3dpp.BackBufferHeight = render_height;
			d3dpp.BackBufferFormat = D3DFMT_A8R8G8B8;
			d3dpp.BackBufferCount = 1;
			d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
			d3dpp.Windowed = TRUE;
			d3dpp.EnableAutoDepthStencil = TRUE;
			d3dpp.AutoDepthStencilFormat = D3DFMT_D24S8;
			d3dpp.Flags = D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
			IDirect3DDevice9* pDeviceTmp = nullptr;
			TRY_D3D(D3D9GlobalServices::Instance->pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, D3D9GlobalServices::Instance->hHostWindow, D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING, &d3dpp, &pDeviceTmp));
			pDevice = pDeviceTmp;
			TRY_D3D(pDevice->Clear(0, nullptr, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x00000000, 1.0, 0));
			IDirect3DSurface9* pSurfaceTmp = nullptr;
			TRY_D3D(pDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &pSurfaceTmp));
			pSurface = pSurfaceTmp;
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
		#pragma endregion
		#pragma region - Section : Managed Interface -
		property System::IntPtr SurfacePtr {
			System::IntPtr get() { return System::IntPtr(pSurface); }
		}
		void BeginScene() {
			TRY_D3D(pDevice->BeginScene());
			TRY_D3D(pDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE));
			TRY_D3D(pDevice->SetRenderState(D3DRS_LIGHTING, FALSE));
			TRY_D3D(pDevice->SetFVF(D3DFVF_XYZW | D3DFVF_DIFFUSE));
		}
		void EndScene() {
			TRY_D3D(pDevice->EndScene());
		}
		void DrawPrimitiveUP(D3DPrimitiveType PrimitiveType, unsigned int PrimitiveCount, System::IntPtr ^pVertexStreamZeroData, unsigned int VertexStreamZeroStride)
		{
			TRY_D3D(pDevice->DrawPrimitiveUP((D3DPRIMITIVETYPE)PrimitiveType, PrimitiveCount, pVertexStreamZeroData->ToPointer(), VertexStreamZeroStride));
		}
		void CopyTo(System::IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride) {
			D3DLOCKED_RECT rectd3d = { 0 };
			RECT rect = { 0, 0, render_width, render_height };
			TRY_D3D(pSurface->LockRect(&rectd3d, &rect, D3DLOCK_READONLY));
			void* copyTo = (void*)bitmap_ptr;
			for (int y = 0; y < render_height; ++y) {
				memcpy((unsigned char*)copyTo + bitmap_stride * y, (unsigned char*)rectd3d.pBits + rectd3d.Pitch * y, 4 * render_width);
			}
			TRY_D3D(pSurface->UnlockRect());
		}
		#pragma endregion
	private:
		IDirect3DDevice9* pDevice = nullptr;
		IDirect3DSurface9* pSurface = nullptr;
	};
}