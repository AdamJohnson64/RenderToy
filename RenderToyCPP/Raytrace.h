////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef RAYTRACE_H
#define RAYTRACE_H

template <typename FLOAT> struct Vector3 { FLOAT x, y, z; };
template <typename FLOAT> struct Vector4 { FLOAT x, y, z, w; };
template <typename FLOAT> struct Matrix44 { FLOAT M[16]; };

template <typename FLOAT>
struct IntersectSimple {
	FLOAT Lambda;
};

template <typename FLOAT>
struct IntersectNormal {
	FLOAT Lambda;
	Vector3<FLOAT> Normal;
};

template <typename FLOAT>
struct IntersectTBN {
	FLOAT Lambda;
	Vector3<FLOAT> Normal;
	Vector3<FLOAT> Tangent;
	Vector3<FLOAT> Bitangent;
};

enum GeometryType {
	GEOMETRY_NONE = 0,
	GEOMETRY_PLANE = 1,
	GEOMETRY_SPHERE = 2,
	GEOMETRY_CUBE = 3,
};

enum MaterialType {
	MATERIAL_NONE = 0,
	MATERIAL_COMMON,
	MATERIAL_CHECKERBOARD_XZ,
};

template <typename FLOAT>
struct MaterialCommon {
	Vector4<FLOAT> Ambient;
	Vector4<FLOAT> Diffuse;
	Vector4<FLOAT> Specular;
	Vector4<FLOAT> Reflect;
	Vector4<FLOAT> Refract;
	FLOAT Ior;
};

template <typename FLOAT>
struct SceneObject {
	Matrix44<FLOAT> Transform;
	Matrix44<FLOAT> TransformInverse;
	GeometryType Geometry;
	int GeometryOffset;
	MaterialType Material;
	int MaterialOffset;
};

template <typename FLOAT>
struct Scene {
	int FileSize;
	int ObjectCount;
	int Reserved0;
	int Reserved1;
	SceneObject<FLOAT> Objects[];
};

#endif