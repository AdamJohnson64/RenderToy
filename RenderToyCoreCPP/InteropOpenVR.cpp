#ifdef OPENVR_INSTALLED
#include "openvr.h"

namespace RenderToy
{
	public enum class Eye
	{
		Left = vr::Eye_Left,
		Right = vr::Eye_Right,
	};
	public value struct HmdMatrix34
	{
		float M11, M21, M31, M41;
		float M12, M22, M32, M42;
		float M13, M23, M33, M43;
	};
	public value struct HmdMatrix44
	{
		float M11, M21, M31, M41;
		float M12, M22, M32, M42;
		float M13, M23, M33, M43;
		float M14, M24, M34, M44;
	};
	public value struct HmdVector3
	{
		float X, Y, Z;
	};
	public enum class TrackedControllerRole
	{
		Invalid = vr::TrackedControllerRole_Invalid,
		LeftHand = vr::TrackedControllerRole_LeftHand,
		RightHand = vr::TrackedControllerRole_RightHand,
		OptOut = vr::TrackedControllerRole_OptOut,
		Max = vr::TrackedControllerRole_Max,
	};
	public enum class TrackingResult
	{
		Uninitialized = vr::TrackingResult_Uninitialized,
		CalibratingInProgress = vr::TrackingResult_Calibrating_InProgress,
		CalibratingOutOfRange = vr::TrackingResult_Calibrating_OutOfRange,
		RunningOk = vr::TrackingResult_Running_OK,
		RunningOutOfRange = vr::TrackingResult_Running_OutOfRange,
	};
	public enum class TrackingUniverseOrigin
	{
		Seated = vr::TrackingUniverseSeated,
		Standing = vr::TrackingUniverseStanding,
		RawAndUncalibrated = vr::TrackingUniverseRawAndUncalibrated,
	};
	public value struct TrackedDevicePose
	{
		HmdMatrix34 mDeviceToAbsoluteTracking;
		HmdVector3 vVelocity;
		HmdVector3 vAngularVelocity;
		TrackingResult eTrackingResult;
		bool bPoseIsValid;
		bool bDeviceIsConnected;
	};
	public ref class OpenVR
	{
	public:
		static OpenVR()
		{
			vrsystem = vr::VR_Init(nullptr, vr::EVRApplicationType::VRApplication_Scene);
			if (vrsystem == nullptr) throw gcnew System::Exception("OpenVR could not be started; check your SteamVR and HMD setup.");
		}
		static TrackedControllerRole GetControllerRoleForTrackedDeviceIndex(vr::TrackedDeviceIndex_t unDeviceIndex)
		{
			return (TrackedControllerRole)vrsystem->GetControllerRoleForTrackedDeviceIndex(unDeviceIndex);
		}
		static void GetDeviceToAbsoluteTrackingPose(TrackingUniverseOrigin eOrigin, float fPredictedSecondsToPhotonsFromNow, cli::array<TrackedDevicePose> ^pTrackedDevicePoseArray)
		{
			pin_ptr<TrackedDevicePose> pTrackedDevicePoseArrayM = &pTrackedDevicePoseArray[0];
			vrsystem->GetDeviceToAbsoluteTrackingPose((vr::ETrackingUniverseOrigin)eOrigin, fPredictedSecondsToPhotonsFromNow, (vr::TrackedDevicePose_t*)&pTrackedDevicePoseArrayM[0], pTrackedDevicePoseArray->Length);
		}
		static HmdMatrix44 GetProjectionMatrix(Eye eEye, float fNearZ, float fFarZ)
		{
			HmdMatrix44 result;
			*reinterpret_cast<vr::HmdMatrix44_t*>(&result) = vrsystem->GetProjectionMatrix((vr::EVREye)eEye, fNearZ, fFarZ);
			return result;
		}
		static HmdMatrix34 GetEyeToHeadTransform(Eye eEye)
		{
			HmdMatrix34 result;
			*reinterpret_cast<vr::HmdMatrix34_t*>(&result) = vrsystem->GetEyeToHeadTransform((vr::EVREye)eEye);
			return result;
		}
		static void GetRecommendedRenderTargetSize(uint32_t %width, uint32_t %height)
		{
			uint32_t vrwidth, vrheight;
			vrsystem->GetRecommendedRenderTargetSize(&vrwidth, &vrheight);
			width = vrwidth;
			height = vrheight;
		}
		static bool GetTimeToPhotons(float %time)
		{
			float fSecondsSinceLastVsync;
			// TODO: Figure out why this value is sometimes 3E+12 (37642607.2 years)
			// This value is outside my lifetime so I can't really confirm it.
			if (!vrsystem->GetTimeSinceLastVsync(&fSecondsSinceLastVsync, nullptr) || fSecondsSinceLastVsync > 1)
			{
				return false;
			}
			float fDisplayFrequency = vrsystem->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_DisplayFrequency_Float);
			float fFrameDuration = 1.f / fDisplayFrequency;
			float fVsyncToPhotons = vrsystem->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SecondsFromVsyncToPhotons_Float);
			// TODO: Figure out why the suggested solution from Valve is incorrect? (see https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetDeviceToAbsoluteTrackingPose)
			// Scan out requires 2 frames to deliver photons to the eyes so we need double the duration.
			float fPredictedSecondsFromNow = fFrameDuration * 2 - fSecondsSinceLastVsync + fVsyncToPhotons;
			time = fPredictedSecondsFromNow;
			return true;
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