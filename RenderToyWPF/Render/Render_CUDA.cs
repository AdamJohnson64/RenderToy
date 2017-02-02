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
            RenderToy.RenderToyCPP.RaycastCPU(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastNormalsCPU(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastNormalsCPU(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastTangentsCPU(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastTangentsCPU(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastBitangentsCPU(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastTangentsCPU(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCPUF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaytraceCPUF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaytraceCPUF64(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastCUDA(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastNormalsCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastNormalsCUDA(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastTangentsCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastTangentsCUDA(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastBitangentsCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaycastBitangentsCUDA(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaytraceCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCUDAF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCPP.RaytraceCUDAF64(SceneFormatter.CreateFlatMemory(scene), SceneFormatter.CreateFlatMemory(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        #endregion
    }
}