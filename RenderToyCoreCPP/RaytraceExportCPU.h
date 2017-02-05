////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// These are the core CPU raytracer entrypoints.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef RAYTRACEEXPORTCPU_H
#define RAYTRACEEXPORTCPU_H

extern "C" void RaycastCPUF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastCPUF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastNormalsCPUF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastNormalsCPUF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastTangentsCPUF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastTangentsCPUF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastBitangentsCPUF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastBitangentsCPUF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCPUF32(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCPUF64(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCPUF64AA(const void* pScene, const void* pMVP, void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int superx, int supery);

#endif  // RAYTRACEEXPORTCPU_H