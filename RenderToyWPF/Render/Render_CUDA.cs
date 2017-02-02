////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RenderToy
{
    public static partial class Render
    {
        #region - Section : Phase 4 - Raytrace Rendering (CUDA) -
        public static bool CUDAAvailable()
        {
            return RenderToy.RenderToyCPP.HaveCUDA();
        }
        public static void RaycastCPU(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastCPU(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastNormalsCPU(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastNormalsCPU(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastTangentsCPU(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastTangentsCPU(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastBitangentsCPU(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastTangentsCPU(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCPUF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaytraceCPUF32(SceneFormatter.CreateFlatMemoryF32(scene), MatrixToFloats(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaytraceCPUF64(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastCUDA(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastNormalsCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastNormalsCUDA(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastTangentsCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastTangentsCUDA(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastBitangentsCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastBitangentsCUDA(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaytraceCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), MatrixToFloats(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCUDAF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaytraceCUDAF64(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        static float[] MatrixToFloats(Matrix3D mvp)
        {
            return new float[16] {
                (float)mvp.M11, (float)mvp.M12, (float)mvp.M13, (float)mvp.M14,
                (float)mvp.M21, (float)mvp.M22, (float)mvp.M23, (float)mvp.M24,
                (float)mvp.M31, (float)mvp.M32, (float)mvp.M33, (float)mvp.M34,
                (float)mvp.M41, (float)mvp.M42, (float)mvp.M43, (float)mvp.M44,
            };
        }
        static double[] MatrixToDoubles(Matrix3D mvp)
        {
            return new double[16] {
                mvp.M11, mvp.M12, mvp.M13, mvp.M14,
                mvp.M21, mvp.M22, mvp.M23, mvp.M24,
                mvp.M31, mvp.M32, mvp.M33, mvp.M34,
                mvp.M41, mvp.M42, mvp.M43, mvp.M44,
            };
        }
        #endregion
    }
}