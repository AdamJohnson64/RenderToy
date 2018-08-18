////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#if OPENVR_INSTALLED
using RenderToy.DirectX;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Transforms;
using System.Linq;
using Valve.VR;

namespace RenderToy
{
    public class OpenVRHelper
    {
        public static Matrix3D ConvertMatrix43(HmdMatrix34_t matrix)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix.m0; m.M21 = matrix.m1; m.M31 = matrix.m2; m.M41 = matrix.m3;
            m.M12 = matrix.m4; m.M22 = matrix.m5; m.M32 = matrix.m6; m.M42 = matrix.m7;
            m.M13 = matrix.m8; m.M23 = matrix.m9; m.M33 = matrix.m10; m.M43 = matrix.m11;
            m.M14 = 0; m.M24 = 0; m.M34 = 0; m.M44 = 1;
            return m;
        }
        public static Matrix3D ConvertMatrix44(HmdMatrix44_t matrix)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix.m0; m.M21 = matrix.m1; m.M31 = matrix.m2; m.M41 = matrix.m3;
            m.M12 = matrix.m4; m.M22 = matrix.m5; m.M32 = matrix.m6; m.M42 = matrix.m7;
            m.M13 = matrix.m8; m.M23 = matrix.m9; m.M33 = matrix.m10; m.M43 = matrix.m11;
            m.M14 = matrix.m12; m.M24 = matrix.m13; m.M34 = matrix.m14; m.M44 = matrix.m15;
            return m;
        }
        public static Matrix3D GetEyeToHeadTransform(EVREye eEye)
        {
            return ConvertMatrix43(System.GetEyeToHeadTransform(eEye));
        }
        public static Matrix3D GetProjectionMatrix(EVREye eEye, float fNear, float fFar)
        {
            return ConvertMatrix44(System.GetProjectionMatrix(eEye, fNear, fFar));
        }
        public static void SubmitLeftHand(Matrix3D hand)
        {
            _lefthand = hand;   
        }
        public static void SubmitRightHand(Matrix3D hand)
        {
            _righthand = hand;
        }
        public static void Update()
        {
            TrackedDevicePose_t[] renderPose = new TrackedDevicePose_t[16];
            TrackedDevicePose_t[] gamePose = new TrackedDevicePose_t[16];
            AGAIN:
            Direct3D11Helper.Dispatcher.Invoke(() =>
            {
                Compositor.WaitGetPoses(renderPose, gamePose);
            });
            float SecondsFromVsyncToPhotons = 0;
            {
                float fSecondsSinceLastVsync = 0;
                ulong pullFrameCounter = 0;
                // TODO: Figure out why this value is sometimes 3E+12 (37642607.2 years)
                // This value is outside my lifetime so I can't really confirm it.
                if (!System.GetTimeSinceLastVsync(ref fSecondsSinceLastVsync, ref pullFrameCounter) || fSecondsSinceLastVsync > 1)
                {
                    goto AGAIN;
                }
                ETrackedPropertyError error = ETrackedPropertyError.TrackedProp_UnknownProperty;
                float fDisplayFrequency = System.GetFloatTrackedDeviceProperty(OpenVR.k_unTrackedDeviceIndex_Hmd, ETrackedDeviceProperty.Prop_DisplayFrequency_Float, ref error);
                float fFrameDuration = 1.0f / fDisplayFrequency;
                float fVsyncToPhotons = System.GetFloatTrackedDeviceProperty(OpenVR.k_unTrackedDeviceIndex_Hmd, ETrackedDeviceProperty.Prop_SecondsFromVsyncToPhotons_Float, ref error);
                // TODO: Figure out why the suggested solution from Valve is incorrect? (see https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetDeviceToAbsoluteTrackingPose)
                // Scan out requires 2 frames to deliver photons to the eyes so we need double the duration.
                SecondsFromVsyncToPhotons = fFrameDuration * 2 - fSecondsSinceLastVsync + fVsyncToPhotons;
            }
            TrackedDevicePose_t[] trackedPoses = new TrackedDevicePose_t[16];
            System.GetDeviceToAbsoluteTrackingPose(ETrackingUniverseOrigin.TrackingUniverseStanding, SecondsFromVsyncToPhotons, trackedPoses);
            {
                _head = transformGLtoDX * MathHelp.Invert(OpenVRHelper.ConvertMatrix43(trackedPoses[0].mDeviceToAbsoluteTracking));
            }
            var hands = trackedPoses
                .Select((pose, index) => new { Pose = pose, Index = index })
                .Where((x) => x.Pose.bDeviceIsConnected && x.Pose.bPoseIsValid && (System.GetControllerRoleForTrackedDeviceIndex((uint)x.Index) == ETrackedControllerRole.RightHand || System.GetControllerRoleForTrackedDeviceIndex((uint)x.Index) == ETrackedControllerRole.LeftHand))
                .ToArray();
            {
                var lefthand = hands.Where(i => System.GetControllerRoleForTrackedDeviceIndex((uint)i.Index) == ETrackedControllerRole.LeftHand).FirstOrDefault();
                if (lefthand != null)
                {
                    _lefthand = OpenVRHelper.ConvertMatrix43(lefthand.Pose.mDeviceToAbsoluteTracking) * transformGLtoDX;
                }
            }
            {
                var righthand = hands.Where(i => System.GetControllerRoleForTrackedDeviceIndex((uint)i.Index) == ETrackedControllerRole.RightHand).FirstOrDefault();
                if (righthand != null)
                {
                    _righthand = OpenVRHelper.ConvertMatrix43(righthand.Pose.mDeviceToAbsoluteTracking) * transformGLtoDX;
                }
            }
        }
        public static Matrix3D _head;
        public static Matrix3D _lefthand;
        public static Matrix3D _righthand;
        public static EVRInitError error;
        public static CVRSystem System = OpenVR.Init(ref error);
        public static CVRCompositor Compositor = OpenVR.Compositor;
        static Matrix3D transformGLtoDX = new Matrix3D(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1);
    }
    public class TransformHMD : ITransform
    {
        public Matrix3D Transform
        {
            get
            {
                return OpenVRHelper._head;
            }
        }
        public OpenVRHelper VRHost { get; protected set; }
    };
    public class TransformLeftHand : ITransform
    {
        public Matrix3D Transform
        {
            get
            {
                return OpenVRHelper._lefthand;
            }
        }
    };
    public class TransformRightHand : ITransform
    {
        public Matrix3D Transform
        {
            get
            {
                return OpenVRHelper._righthand;
            }
        }
    };
    public class MaterialOpenVRCameraDistorted : IMaterial
    {
        public MaterialOpenVRCameraDistorted(OpenVRHelper vrhost, CVRTrackedCamera camera)
        {
            TrackedCamera = camera;
            ulong handle = 0;
            TrackedCamera.AcquireVideoStreamingService(0, ref handle);
            TrackedCameraHandle = handle;
        }
        public bool IsConstant()
        {
            return false;
        }
        public CVRTrackedCamera TrackedCamera { get; protected set; }
        public ulong TrackedCameraHandle { get; protected set; }
    };
}
#endif // OPENVR_INSTALLED