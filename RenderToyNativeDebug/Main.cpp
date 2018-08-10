#include <amp.h>
#include <amp_graphics.h>
#include <amp_math.h>
#include <d3d11.h>
#include <d3d12.h>

#define TRY_D3D(FUNCTION) if (FUNCTION != S_OK) DebugBreak();

void main()
{
	ID3D11Device *ppDevice = nullptr;
	D3D_FEATURE_LEVEL featurelevel = D3D_FEATURE_LEVEL_11_1;
	TRY_D3D(::D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &featurelevel, 1, D3D11_SDK_VERSION, &ppDevice, nullptr, nullptr));
	void *ppDevice12 = nullptr;
	TRY_D3D(::D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device3), &ppDevice12));
	auto accelerator_view = concurrency::direct3d::create_accelerator_view(ppDevice);
}