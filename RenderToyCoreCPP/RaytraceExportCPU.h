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

extern "C" void RaycastCPU(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastNormalsCPU(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastTangentsCPU(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaycastBitangentsCPU(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCPUF32(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void RaytraceCPUF64(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

#endif  // RAYTRACEEXPORTCPU_H