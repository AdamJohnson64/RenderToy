#ifdef OPENVR_INSTALLED
#include "openvr.h"

namespace RenderToy
{
	public enum class Eye
	{
		Left = vr::Eye_Left,
		Right = vr::Eye_Right,
	};
	public enum class TrackedControllerRole
	{
		Invalid = vr::TrackedControllerRole_Invalid,
		LeftHand = vr::TrackedControllerRole_LeftHand,
		RightHand = vr::TrackedControllerRole_RightHand,
		OptOut = vr::TrackedControllerRole_OptOut,
		Max = vr::TrackedControllerRole_Max,
	};
	public ref class OpenVR
	{
	private:
		static void ConvertMatrix(cli::array<float> ^matrixOut, const vr::HmdMatrix34_t &matrixIn)
		{
			for (int j = 0; j < 3; ++j)
			{
				for (int i = 0; i < 4; ++i)
				{
					matrixOut[i + j * 4] = ((float*)&matrixIn)[i + j * 4];
				}
			}
		}
	public:
		static OpenVR()
		{
			vrsystem = vr::VR_Init(nullptr, vr::EVRApplicationType::VRApplication_Scene);
		}
		static void GetRecommendedRenderTargetSize(uint32_t %width, uint32_t %height)
		{
			uint32_t vrwidth, vrheight;
			vrsystem->GetRecommendedRenderTargetSize(&vrwidth, &vrheight);
			width = vrwidth;
			height = vrheight;
		}
		static bool LocateDeviceId(cli::array<float> ^matrix43, int deviceID)
		{
			if (matrix43 == nullptr) return false;
			vr::TrackedDevicePose_t trackedDevicePoseArray[16];
			vrsystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseSeated, 0, trackedDevicePoseArray, 16);
			if (vrsystem->IsTrackedDeviceConnected(deviceID))
			{
				ConvertMatrix(matrix43, trackedDevicePoseArray[deviceID].mDeviceToAbsoluteTracking);
				return true;
			}
			return false;
		}
		static bool LocateDeviceRole(cli::array<float> ^matrix43, TrackedControllerRole deviceRole)
		{
			if (matrix43 == nullptr) return false;
			vr::TrackedDevicePose_t trackedDevicePoseArray[16];
			vrsystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseSeated, 0, trackedDevicePoseArray, 16);
			for (int device = vr::k_unTrackedDeviceIndex_Hmd; device < vr::k_unMaxTrackedDeviceCount; ++device)
			{
				if (vrsystem->IsTrackedDeviceConnected(device))
				{
					if (vrsystem->GetControllerRoleForTrackedDeviceIndex(device) == (vr::ETrackedControllerRole)deviceRole)
					{
						ConvertMatrix(matrix43, trackedDevicePoseArray[device].mDeviceToAbsoluteTracking);
						return true;
					}
				}
			}
			return false;
		}
	public:
		static vr::IVRSystem *vrsystem;
	};

	public ref class OpenVRCompositor
	{
	public:
		static OpenVRCompositor()
		{
			vrcompositor = vr::VRCompositor();
		}
		static void WaitGetPoses()
		{
			vr::TrackedDevicePose_t poserender, posegame;
			vrcompositor->WaitGetPoses(&poserender, 1, &posegame, 1);
		}
		static void Submit(Eye eEye, System::IntPtr pTexture)
		{
			vr::Texture_t texture;
			texture.handle = pTexture.ToPointer();
			texture.eType = vr::TextureType_DirectX;
			texture.eColorSpace = vr::ColorSpace_Auto;
			vrcompositor->Submit((vr::EVREye)eEye, &texture, nullptr);
		}
	private:
		static vr::IVRCompositor *vrcompositor;
	};
}
#endif // OPENVR_INSTALLED