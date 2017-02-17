////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// These are the core CUDA raytracer entrypoints.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef RAYTRACEEXPORTCUDA_H
#define RAYTRACEEXPORTCUDA_H

extern "C" bool HaveCUDA();
extern "C" void RaycastCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastNormalsCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastNormalsCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastTangentsCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastTangentsCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastBitangentsCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaycastBitangentsCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaytraceCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void RaytraceCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride);
extern "C" void AmbientOcclusionCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int hemisample_count, const void* hemisamples);
extern "C" void AmbientOcclusionCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int hemisample_count, const void* hemisamples);
extern "C" void AmbientOcclusionMPCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, int hemisample_count, const void* hemisamples);
extern "C" void AmbientOcclusionFMPCUDAF32(const void* pScene, const void* pMVP, void* acc_ptr, int render_width, int render_height, int bitmap_stride, int hemisample_count, const void* hemisamples);
extern "C" void ToneMap(const void* acc_ptr, int acc_stride, void* bitmap_ptr, int render_width, int render_height, int bitmap_stride, float rescale);

#endif  // RAYTRACEEXPORTCUDA_H