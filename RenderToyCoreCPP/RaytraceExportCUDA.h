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
extern "C" void RaycastCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastNormalsCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastNormalsCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastTangentsCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastTangentsCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastBitangentsCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastBitangentsCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCUDAF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCUDAF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

#endif  // RAYTRACEEXPORTCUDA_H