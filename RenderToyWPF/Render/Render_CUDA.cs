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
        public static bool CUDAAvailable()
        {
            return RenderToy.RenderToyCLI.HaveCUDA();
        }
        #region - Section : Raycast -
        public static void RaycastCPUF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastCPUF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastCPUF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastCUDAF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastCUDAF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        #endregion
        #region - Section : Raycast Normals -
        public static void RaycastNormalsCPUF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastNormalsCPUF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastNormalsCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastNormalsCPUF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastNormalsCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastNormalsCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastNormalsCUDAF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastNormalsCUDAF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        #endregion
        #region - Section : Raycast Tangents -
        public static void RaycastTangentsCPUF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastTangentsCPUF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastTangentsCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastTangentsCPUF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastTangentsCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastTangentsCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastTangentsCUDAF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastTangentsCUDAF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        #endregion
        #region - Section : Raycast Bitangents -
        public static void RaycastBitangentsCPUF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastTangentsCPUF32(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastBitangentsCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastTangentsCPUF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastBitangentsCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastBitangentsCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaycastBitangentsCUDAF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaycastBitangentsCUDAF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        #endregion
        #region - Section : Raytrace -
        public static void RaytraceCPUF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaytraceCPUF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCPUF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaytraceCPUF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaytraceCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCUDAF64(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RenderToyCLI.RaytraceCUDAF64(SceneFormatter.CreateFlatMemoryF64(scene), SceneFormatter.CreateFlatMemoryF64(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        #endregion
    }
}