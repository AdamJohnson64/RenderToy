////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef RENDERTOY_H
#define RENDERTOY_H

extern "C" bool HaveCUDA();
extern "C" void CPURaycast(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaycastNormals(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaycastTangents(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPURaycastBitangents(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPUF32Raytrace(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CPUF64Raytrace(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycast(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycastNormals(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycastTangents(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDARaycastBitangents(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDAF32Raytrace(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);
extern "C" void CUDAF64Raytrace(void *pScene, void *pMVP, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride);

#endif  // RENDERTOY_H