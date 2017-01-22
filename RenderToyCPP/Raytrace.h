#pragma once
#ifndef RAYTRACE_H
#define RAYTRACE_H

struct Matrix4D {
	double M[16];
};

struct IntersectSimple {
	double Lambda;
};

struct IntersectNormal {
	double Lambda;
	double3 Normal;
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

struct MaterialCommon {
	double4 Ambient;
	double4 Diffuse;
	double4 Specular;
	double4 Reflect;
	double4 Refract;
	double Ior;
};

struct SceneObject {
	Matrix4D Transform;
	Matrix4D TransformInverse;
	GeometryType Geometry;
	int GeometryOffset;
	MaterialType Material;
	int MaterialOffset;
};

struct Scene {
	int FileSize;
	int ObjectCount;
	SceneObject Objects[];
};

#endif