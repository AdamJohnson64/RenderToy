////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#if OPENVR_INSTALLED
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Transforms;
using System.Linq;

namespace RenderToy
{
    public class OpenVRHelper
    {
        public static Matrix3D ConvertMatrix43(HmdMatrix34 matrix)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix.M11; m.M21 = matrix.M21; m.M31 = matrix.M31; m.M41 = matrix.M41;
            m.M12 = matrix.M12; m.M22 = matrix.M22; m.M32 = matrix.M32; m.M42 = matrix.M42;
            m.M13 = matrix.M13; m.M23 = matrix.M23; m.M33 = matrix.M33; m.M43 = matrix.M43;
            m.M14 = 0; m.M24 = 0; m.M34 = 0; m.M44 = 1;
            return m;
        }
        public static Matrix3D ConvertMatrix44(HmdMatrix44 matrix)
        {
            Matrix3D m = new Matrix3D();
            m.M11 = matrix.M11; m.M21 = matrix.M21; m.M31 = matrix.M31; m.M41 = matrix.M41;
            m.M12 = matrix.M12; m.M22 = matrix.M22; m.M32 = matrix.M32; m.M42 = matrix.M42;
            m.M13 = matrix.M13; m.M23 = matrix.M23; m.M33 = matrix.M33; m.M43 = matrix.M43;
            m.M14 = matrix.M14; m.M24 = matrix.M24; m.M34 = matrix.M34; m.M44 = matrix.M44;
            return m;
        }
        public Matrix3D GetEyeToHeadTransform(Eye eEye)
        {
            return ConvertMatrix43(System.GetEyeToHeadTransform(eEye));
        }
        public Matrix3D GetProjectionMatrix(Eye eEye, float fNear, float fFar)
        {
            return ConvertMatrix44(System.GetProjectionMatrix(eEye, fNear, fFar));
        }
        public void SubmitLeftHand(Matrix3D hand)
        {
            
        }
        public void SubmitRightHand(Matrix3D hand)
        {
            _righthand = hand;
        }
        public void Update()
        {
            Compositor.WaitGetPoses();
            float fPredictedSecondsToPhotonsFromNow = 0;
            while (!System.GetTimeToPhotons(ref fPredictedSecondsToPhotonsFromNow))
            {
                Compositor.WaitGetPoses();
            }
            TrackedDevicePose[] poses = new TrackedDevicePose[16];
            System.GetDeviceToAbsoluteTrackingPose(TrackingUniverseOrigin.Standing, fPredictedSecondsToPhotonsFromNow, poses);
            {
                _head = transformGLtoDX * MathHelp.Invert(OpenVRHelper.ConvertMatrix43(poses[0].mDeviceToAbsoluteTracking));
            }
            var hands = poses
                .Select((pose, index) => new { Pose = pose, Index = index })
                .Where((x) => x.Pose.bDeviceIsConnected && x.Pose.bPoseIsValid && (System.GetControllerRoleForTrackedDeviceIndex((uint)x.Index) == TrackedControllerRole.RightHand || System.GetControllerRoleForTrackedDeviceIndex((uint)x.Index) == TrackedControllerRole.LeftHand))
                .ToArray();
            {
                var lefthand = hands.Where(i => System.GetControllerRoleForTrackedDeviceIndex((uint)i.Index) == TrackedControllerRole.LeftHand).FirstOrDefault();
                if (lefthand != null)
                {
                    _lefthand = OpenVRHelper.ConvertMatrix43(lefthand.Pose.mDeviceToAbsoluteTracking) * transformGLtoDX;
                }
            }
            {
                var righthand = hands.Where(i => System.GetControllerRoleForTrackedDeviceIndex((uint)i.Index) == TrackedControllerRole.RightHand).FirstOrDefault();
                if (righthand != null)
                {
                    _righthand = OpenVRHelper.ConvertMatrix43(righthand.Pose.mDeviceToAbsoluteTracking) * transformGLtoDX;
                }
            }
        }
        public Matrix3D _head;
        public Matrix3D _lefthand;
        public Matrix3D _righthand;
        public VRSystem System = new VRSystem();
        public OpenVRCompositor Compositor = new OpenVRCompositor();
        static Matrix3D transformGLtoDX = new Matrix3D(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1);
    }
    public interface IVRHost
    {
        OpenVRHelper VRHost { get; }
    }
    public class TransformHMD : ITransform, IVRHost
    {
        public TransformHMD(OpenVRHelper vrhost)
        {
            VRHost = vrhost;
        }
        public Matrix3D Transform
        {
            get
            {
                return VRHost._head;
            }
        }
        public OpenVRHelper VRHost { get; protected set; }
    };
    public class TransformLeftHand : ITransform, IVRHost
    {
        public TransformLeftHand(OpenVRHelper vrhost)
        {
            VRHost = vrhost;
        }
        public Matrix3D Transform
        {
            get
            {
                return VRHost._lefthand;
            }
        }
        public OpenVRHelper VRHost { get; protected set; }
    };
    public class TransformRightHand : ITransform, IVRHost
    {
        public TransformRightHand(OpenVRHelper vrhost)
        {
            VRHost = vrhost;
        }
        public Matrix3D Transform
        {
            get
            {
                return VRHost._righthand;
            }
        }
        public OpenVRHelper VRHost { get; protected set; }
    };
    public class MaterialOpenVRCameraDistorted : IMaterial, IVRHost
    {
        public MaterialOpenVRCameraDistorted(OpenVRHelper vrhost, VRTrackedCamera camera)
        {
            VRHost = vrhost;
            TrackedCamera = camera;
            TrackedCameraHandle = TrackedCamera.AcquireVideoStreamingService(0);
        }
        public bool IsConstant()
        {
            return false;
        }
        public OpenVRHelper VRHost { get; protected set; }
        public VRTrackedCamera TrackedCamera { get; protected set; }
        public ulong TrackedCameraHandle { get; protected set; }
    };
}
#endif // OPENVR_INSTALLED