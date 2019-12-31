#include <dxgi1_6.h>
#include "InteropCommon.h"

#define TRY(FUNCTION) if (FUNCTION != S_OK) throw gcnew System::Exception(#FUNCTION);

using namespace System::Runtime::InteropServices;

namespace RenderToy
{
#pragma region - DXGI Global Functions -
	public ref class DXGI
	{
	public:
		static RenderToyCOM::IDXGIFactory2^ CreateDXGIFactory2()
		{
			void *ppFactory = nullptr;
			TRY(::CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, __uuidof(IDXGIFactory2), &ppFactory));
			return (RenderToyCOM::IDXGIFactory2^)Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppFactory), RenderToyCOM::IDXGIFactory2::typeid);
		}
		// BUGFIX: Due to an issue in IDXGIOutputDuplication we can't wrap it in an RCW to send to .Net.
		// To work around this we pass the bare interface pointer around which is obviously dangerous:
		// - It's not type safe at all.
		// - It won't Release() properly without special treatment.
		// See here:
		// https://social.msdn.microsoft.com/Forums/en-US/028b6348-6484-4906-94f2-5d9980810c64/queryinterface-for-iunknown-fails-for-instance-of-idxgioutputduplication
		static System::IntPtr IDXGIOutput1_DuplicateOutput(RenderToyCOM::IDXGIOutput1 ^output, RenderToyCOM::ID3D11Device ^device)
		{
			auto com_device = Marshal::GetIUnknownForObject(device);
			IUnknown *pDevice = (IUnknown*)com_device.ToPointer();
			auto com_output1 = Marshal::GetComInterfaceForObject(output, RenderToyCOM::IDXGIOutput1::typeid);
			IDXGIOutput1 *pOutput1 = (IDXGIOutput1*)com_output1.ToPointer();
			IDXGIOutputDuplication *ppOutputDuplication = nullptr;
			TRY(pOutput1->DuplicateOutput(pDevice, &ppOutputDuplication));
            Marshal::Release(com_output1);
			return System::IntPtr(ppOutputDuplication);
		}
		static RenderToyCOM::IDXGIResource^ IDXGIOutputDuplication_AcquireNextFrame(System::IntPtr outputduplication, UINT TimeoutInMilliseconds, RenderToyCOM::DXGI_OUTDUPL_FRAME_INFO %ppFrameInfo)
		{
			DXGI_OUTDUPL_FRAME_INFO FrameInfo = { 0 };
			IDXGIResource *ppDesktopResource = nullptr;
			IDXGIOutputDuplication *ppOutputDuplication = (IDXGIOutputDuplication*)outputduplication.ToPointer();
			// NOTE: DXGI_ERROR_WAIT_TIMEOUT is the only acceptable error here.
			// Everything else is an exception.
			auto error = ppOutputDuplication->AcquireNextFrame(TimeoutInMilliseconds, &FrameInfo, &ppDesktopResource);
			if (error == DXGI_ERROR_WAIT_TIMEOUT) return nullptr;
			TRY(error);
			auto comiface = Marshal::GetTypedObjectForIUnknown(System::IntPtr(ppDesktopResource), RenderToyCOM::IDXGIResource::typeid);
			ppFrameInfo.LastPresentTime.QuadPart = FrameInfo.LastPresentTime.QuadPart;
			ppFrameInfo.LastMouseUpdateTime.QuadPart = FrameInfo.LastMouseUpdateTime.QuadPart;
			ppFrameInfo.AccumulatedFrames = FrameInfo.AccumulatedFrames;
			ppFrameInfo.RectsCoalesced = FrameInfo.RectsCoalesced;
			ppFrameInfo.ProtectedContentMaskedOut = FrameInfo.ProtectedContentMaskedOut;
			ppFrameInfo.PointerPosition.Position.X = FrameInfo.PointerPosition.Position.x;
			ppFrameInfo.PointerPosition.Position.Y = FrameInfo.PointerPosition.Position.y;
			ppFrameInfo.PointerPosition.Visible = FrameInfo.PointerPosition.Visible;
			ppFrameInfo.TotalMetadataBufferSize = FrameInfo.TotalMetadataBufferSize;
			ppFrameInfo.PointerShapeBufferSize = FrameInfo.PointerShapeBufferSize;
			return (RenderToyCOM::IDXGIResource^)comiface;
		}
		static void IDXGIOutputDuplication_ReleaseFrame(System::IntPtr outputduplication)
		{
			IDXGIResource *ppDesktopResource = nullptr;
			IDXGIOutputDuplication *ppOutputDuplication = (IDXGIOutputDuplication*)outputduplication.ToPointer();
			TRY(ppOutputDuplication->ReleaseFrame());
		}
	};
#pragma endregion
}