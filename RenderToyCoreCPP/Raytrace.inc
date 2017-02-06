////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This file contains the common parts of the raytracer shared by CPU, AMP and
// CUDA implementations.
//
// All parts of this module are templated over their floating point type to
// support both IEEE 754 32-bit float and 64-bit double implementation.
////////////////////////////////////////////////////////////////////////////////

// Type-safe Basic Math.
template <typename FLOAT>
struct FloatTraits {
};

template <>
struct FloatTraits<float> {
	constexpr static float Max = FLT_MAX;
};

template <>
struct FloatTraits<double> {
	constexpr static double Max = DBL_MAX;
};

DEVICE_PREFIX float Power(float v, float p) DEVICE_SUFFIX { return powf(v, p); }
DEVICE_PREFIX double Power(double v, double p) DEVICE_SUFFIX { return pow(v, p); }
DEVICE_PREFIX float SquareRoot(float v) DEVICE_SUFFIX { return sqrtf(v); }
DEVICE_PREFIX double SquareRoot(double v) DEVICE_SUFFIX { return sqrt(v); }

// Vector construction.
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> make_vector3(FLOAT x, FLOAT y, FLOAT z) DEVICE_SUFFIX { Vector3<FLOAT> o; o.x = x; o.y = y; o.z = z; return o; }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> make_vector4(FLOAT x, FLOAT y, FLOAT z, FLOAT w) DEVICE_SUFFIX { Vector4<FLOAT> o; o.x = x; o.y = y; o.z = z; o.w = w; return o; }

// Basic Math Primitives.
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator-(const Vector3<FLOAT>& val) DEVICE_SUFFIX { return make_vector3<FLOAT>(-val.x, -val.y, -val.z); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator-(const Vector4<FLOAT>& val) DEVICE_SUFFIX { return make_vector4<FLOAT>(-val.z, -val.y, -val.z, -val.w); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator+(const Vector3<FLOAT>& lhs, const Vector3<FLOAT>& rhs) DEVICE_SUFFIX { return make_vector3<FLOAT>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator+(const Vector4<FLOAT>& lhs, const Vector4<FLOAT>& rhs) DEVICE_SUFFIX { return make_vector4<FLOAT>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator-(const Vector3<FLOAT>& lhs, const Vector3<FLOAT>& rhs) DEVICE_SUFFIX { return make_vector3<FLOAT>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator-(const Vector4<FLOAT>& lhs, const Vector4<FLOAT>& rhs) DEVICE_SUFFIX { return make_vector4<FLOAT>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator*(const Vector3<FLOAT>& lhs, FLOAT rhs) DEVICE_SUFFIX { return make_vector3<FLOAT>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator*(FLOAT lhs, const Vector3<FLOAT>& rhs) DEVICE_SUFFIX { return rhs * lhs; }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator*(const Vector4<FLOAT>& lhs, FLOAT rhs) DEVICE_SUFFIX { return make_vector4<FLOAT>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator*(FLOAT lhs, const Vector4<FLOAT>& rhs) DEVICE_SUFFIX { return rhs * lhs; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> operator/(const Vector3<FLOAT>& lhs, FLOAT rhs) DEVICE_SUFFIX { return lhs * (1 / rhs); }
template <typename FLOAT> DEVICE_PREFIX Vector4<FLOAT> operator/(const Vector4<FLOAT>& lhs, FLOAT rhs) DEVICE_SUFFIX { return lhs * (1 / rhs); }

// Common Math Primitives.
template <typename FLOAT> DEVICE_PREFIX FLOAT Clamp(FLOAT min, FLOAT max, FLOAT val) DEVICE_SUFFIX { return val < min ? min : (val > max ? max : val); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> Cross(const Vector3<FLOAT>& lhs, const Vector3<FLOAT> rhs) DEVICE_SUFFIX { return make_vector3<FLOAT>(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x); }
template <typename FLOAT> DEVICE_PREFIX FLOAT Dot(const Vector3<FLOAT>& lhs, const Vector3<FLOAT>& rhs) DEVICE_SUFFIX { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
template <typename FLOAT> DEVICE_PREFIX FLOAT Length(FLOAT x, FLOAT y, FLOAT z) DEVICE_SUFFIX { return SquareRoot(x * x + y * y + z * z); }
template <typename FLOAT> DEVICE_PREFIX FLOAT Lerp(FLOAT y1, FLOAT y2, FLOAT x) DEVICE_SUFFIX { return y1 + (y2 - y1) * x; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> Normalize(const Vector3<FLOAT>& val) DEVICE_SUFFIX { return val / Length(val.x, val.y, val.z); }

// Matrix Math.
template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> TransformPoint(const Matrix44<FLOAT>& m, const Vector3<FLOAT>& p) DEVICE_SUFFIX {
	return make_vector3<FLOAT>(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z + m.M[12],
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z + m.M[13],
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z + m.M[14]);
}

template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> TransformVector(const Matrix44<FLOAT>& m, const Vector3<FLOAT>& p) DEVICE_SUFFIX {
	return make_vector3<FLOAT>(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z,
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z,
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z);
}

template <typename FLOAT>
DEVICE_PREFIX Vector4<FLOAT> Transform(const Matrix44<FLOAT>& m, const Vector4<FLOAT>& p) DEVICE_SUFFIX {
	return make_vector4<FLOAT>(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z + m.M[12] * p.w,
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z + m.M[13] * p.w,
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z + m.M[14] * p.w,
		m.M[3] * p.x + m.M[7] * p.y + m.M[11] * p.z + m.M[15] * p.w);
}

template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> Reflect(const Vector3<FLOAT>& incident, const Vector3<FLOAT>& normal) DEVICE_SUFFIX {
	return incident - 2 * Dot(incident, normal) * normal;
}

template <typename FLOAT>
DEVICE_PREFIX Vector3<FLOAT> Refract(const Vector3<FLOAT>& incident, const Vector3<FLOAT>& normal, FLOAT ior) DEVICE_SUFFIX {
	FLOAT cosi = Clamp<FLOAT>(-1, 1, Dot(incident, normal));
	FLOAT etai = 1, etat = ior;
	Vector3<FLOAT> n = normal;
	if (cosi < 0) { cosi = -cosi; }
	else { FLOAT tmp = etai; etai = etat; etat = tmp; n = -normal; }
	FLOAT eta = etai / etat;
	FLOAT k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? make_vector3<FLOAT>(0, 0, 0) : (eta * incident + (eta * cosi - SquareRoot(k)) * n);
}

template <typename FLOAT>
DEVICE_PREFIX FLOAT SchlickApprox(const Vector3<FLOAT>& incident, const Vector3<FLOAT>& normal, FLOAT ior1, FLOAT ior2) DEVICE_SUFFIX {
	FLOAT coeff = (ior1 - ior2) / (ior1 + ior2);
	coeff = coeff * coeff;
	return coeff + (1 - coeff) * Power((1 - Dot(-incident, normal)), 5);
}

// Geometric Math.
template <typename FLOAT>
DEVICE_PREFIX FLOAT IntersectPlane(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const Vector3<FLOAT>& plane_normal, FLOAT plane_distance) DEVICE_SUFFIX {
	return (plane_distance - Dot(plane_normal, origin)) / Dot(plane_normal, direction);
}

template <typename FLOAT, typename RAYQUERY> DEVICE_PREFIX void SetLambda(RAYQUERY& result, FLOAT value) DEVICE_SUFFIX { result.Lambda = value; }
template <typename FLOAT> DEVICE_PREFIX void SetNormal(IntersectSimple<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetNormal(IntersectNormal<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX { result.Normal = value; }
template <typename FLOAT> DEVICE_PREFIX void SetNormal(IntersectTBN<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX { result.Normal = value; }
template <typename FLOAT> DEVICE_PREFIX void SetTangent(IntersectSimple<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetTangent(IntersectNormal<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetTangent(IntersectTBN<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX { result.Tangent = value; }
template <typename FLOAT> DEVICE_PREFIX void SetBitangent(IntersectSimple<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetBitangent(IntersectNormal<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX {}
template <typename FLOAT> DEVICE_PREFIX void SetBitangent(IntersectTBN<FLOAT>& result, Vector3<FLOAT> value) DEVICE_SUFFIX { result.Bitangent = value; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetNormal(const IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetNormal(const IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return result.Normal; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetNormal(const IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return result.Normal; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetTangent(const IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetTangent(const IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetTangent(const IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return result.Tangent; }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetBitangent(const IntersectSimple<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetBitangent(const IntersectNormal<FLOAT>& result) DEVICE_SUFFIX { return make_vector3<FLOAT>(0, 0, 0); }
template <typename FLOAT> DEVICE_PREFIX Vector3<FLOAT> GetBitangent(const IntersectTBN<FLOAT>& result) DEVICE_SUFFIX { return result.Bitangent; }

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX RAYQUERY IntersectCube(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	RAYQUERY result;
	Vector3<FLOAT> face_normal[] = {
		make_vector3<FLOAT>(-1,0,0), make_vector3<FLOAT>(+1,0,0),
		make_vector3<FLOAT>(0,-1,0), make_vector3<FLOAT>(0,+1,0),
		make_vector3<FLOAT>(0,0,-1), make_vector3<FLOAT>(0,0,+1),
	};
	Vector3<FLOAT> face_tangent[] = {
		make_vector3<FLOAT>(0,0,-1), make_vector3<FLOAT>(0,0,+1),
		make_vector3<FLOAT>(+1,0,0), make_vector3<FLOAT>(+1,0,0),
		make_vector3<FLOAT>(+1,0,0), make_vector3<FLOAT>(-1,0,0),
	
	};
	Vector3<FLOAT> face_bitangent[] = {
		make_vector3<FLOAT>(0,-1,0), make_vector3<FLOAT>(0,-1,0),
		make_vector3<FLOAT>(0,0,+1), make_vector3<FLOAT>(0,0,-1),
		make_vector3<FLOAT>(0,-1,0), make_vector3<FLOAT>(0,-1,0),
	};
	FLOAT best_lambda = FloatTraits<FLOAT>::Max;
	for (int face_index = 0; face_index < 6; ++face_index) {
		FLOAT lambda = IntersectPlane<FLOAT>(origin, direction, face_normal[face_index], 1);
		if (lambda < 0 || lambda > best_lambda) continue;
		Vector3<FLOAT> point = origin + lambda * direction;
		// Check that the point is inside every other plane.
		bool use_face = true;
		for (int check_face = 0; check_face < 6; ++check_face) {
			if (face_index == check_face) continue;
			FLOAT inside = Dot(point, face_normal[check_face]) - 1;
			if (inside > 0) {
				use_face = false;
				break;
			}
		}
		if (use_face) {
			best_lambda = lambda;
			SetNormal(result, face_normal[face_index]);
			SetTangent(result, face_tangent[face_index]);
			SetBitangent(result, face_bitangent[face_index]);
		}
	}
	SetLambda(result, best_lambda);
	return result;
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX RAYQUERY IntersectPlane(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	RAYQUERY result;
	const FLOAT PLANE_DISTANCE = 0;
	const Vector3<FLOAT> PLANE_NORMAL = make_vector3<FLOAT>(0, 1, 0);
	SetLambda(result, (PLANE_DISTANCE - Dot(PLANE_NORMAL, origin)) / Dot(PLANE_NORMAL, direction));
	SetNormal(result, PLANE_NORMAL);
	SetTangent(result, make_vector3<FLOAT>(1, 0, 0));
	SetBitangent(result, make_vector3<FLOAT>(0, 0, 1));
	return result;
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX RAYQUERY IntersectSphere(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	RAYQUERY result;
	const FLOAT SPHERE_RADIUS = 1;
	FLOAT a = Dot(direction, direction);
	FLOAT b = 2 * Dot(origin, direction);
	FLOAT c = Dot(origin, origin) - SPHERE_RADIUS * SPHERE_RADIUS;
	// If the determinant is negative then there are no real roots and this will be NaN.
	FLOAT det = SquareRoot(b * b - 4 * a * c);
	// "a" cannot be negative so (worst case) these lambdas are +Inf.
	FLOAT den = 2 * a;
	FLOAT lambda1 = (-b - det) / den;
	FLOAT lambda2 = (-b + det) / den;
	FLOAT lambda_best = FloatTraits<FLOAT>::Max;
	if (lambda1 >= 0 && lambda1 < lambda_best) lambda_best = lambda1;
	if (lambda2 >= 0 && lambda2 < lambda_best) lambda_best = lambda2;
	// Construct a tangent basis at this point.
	Vector3<FLOAT> vector_normal = Normalize(origin + lambda_best * direction);
	Vector3<FLOAT> vector_bitangent = make_vector3<FLOAT>(0,-1,0);
	Vector3<FLOAT> vector_tangent = Normalize(Cross(vector_bitangent, vector_normal));
	vector_bitangent = Normalize(Cross(vector_normal, vector_bitangent));
	SetLambda(result, lambda_best);
	SetNormal(result, vector_normal);
	SetTangent(result, vector_tangent);
	SetBitangent(result, vector_bitangent);
	// TODO: Implement tangent and bitangent.
	return result;
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX RAYQUERY Intersect(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, GeometryType geometry) DEVICE_SUFFIX {
	switch (geometry) {
	case GEOMETRY_PLANE:
		return IntersectPlane<FLOAT, RAYQUERY>(origin, direction);
	case GEOMETRY_SPHERE:
		return IntersectSphere<FLOAT, RAYQUERY>(origin, direction);
	case GEOMETRY_CUBE:
		return IntersectCube<FLOAT, RAYQUERY>(origin, direction);
	default:
		RAYQUERY result;
		SetLambda(result, FloatTraits<FLOAT>::Max);
		return result;
	}
}

template <typename FLOAT, typename RAYQUERY>
DEVICE_PREFIX RAYQUERY Intersect(const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction, const SceneObject<FLOAT>& scene_object) DEVICE_SUFFIX {
	Vector3<FLOAT> transformed_origin = TransformPoint(scene_object.TransformInverse, origin);
	Vector3<FLOAT> transformed_direction = TransformVector(scene_object.TransformInverse, direction);
	RAYQUERY result = Intersect<FLOAT, RAYQUERY>(transformed_origin, transformed_direction, scene_object.Geometry);
	SetNormal(result, TransformVector(scene_object.Transform, GetNormal(result)));
	SetTangent(result, TransformVector(scene_object.Transform, GetTangent(result)));
	SetBitangent(result, TransformVector(scene_object.Transform, GetBitangent(result)));
	return result;
}

template <typename FLOAT>
DEVICE_PREFIX IntersectObject<FLOAT> Intersect(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	FLOAT best_lambda = FloatTraits<FLOAT>::Max;
	const SceneObject<FLOAT>* best_object = nullptr;
	for (int i = 0; i < pScene.ObjectCount; ++i) {
		const SceneObject<FLOAT>& scene_object = pScene.Objects[i];
		FLOAT lambda = Intersect<FLOAT, IntersectSimple<FLOAT>>(origin, direction, scene_object).Lambda;
		if (lambda >= 0 && lambda < best_lambda) {
			best_lambda = lambda;
			best_object = &scene_object;
		}
	}
	IntersectObject<FLOAT> result;
	result.Lambda = best_lambda;
	if (best_object != nullptr) {
		result.Object = *best_object;
	}
	return result;
}

// Raytracer Ray Tests.
template <typename FLOAT>
DEVICE_PREFIX bool RayShadow(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	for (int i = 0; i < pScene.ObjectCount; ++i) {
		const SceneObject<FLOAT>& scene_object = pScene.Objects[i];
		Vector3<FLOAT> transformed_origin = TransformPoint(scene_object.TransformInverse, origin);
		Vector3<FLOAT> transformed_direction = TransformVector(scene_object.TransformInverse, direction);
		FLOAT lambda = Intersect<FLOAT, IntersectSimple<FLOAT>>(transformed_origin, transformed_direction, scene_object.Geometry).Lambda;
		if (lambda >= 0 && lambda < FloatTraits<FLOAT>::Max) return true;
	}
	return false;
}

template <typename FLOAT>
DEVICE_PREFIX Vector4<FLOAT> Raycast(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	auto intersect = Intersect(pScene, origin, direction);
	if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
		return make_vector4<FLOAT>(0, 0, 0, 0);
	}
	Vector3<FLOAT> vector_point = origin + intersect.Lambda * direction;
	switch (intersect.Object.Material) {
	case MATERIAL_COMMON:
	{
		if (intersect.Object.MaterialOffset != 0) {
			const MaterialCommon<FLOAT>* pMaterial = (const MaterialCommon<FLOAT>*)((const int*)&pScene + intersect.Object.MaterialOffset / 4);
			return pMaterial->Diffuse;
		}
	}
	break;
	case MATERIAL_CHECKERBOARD_XZ:
	{
		int mx = (vector_point.x - floor(vector_point.x)) < 0.5 ? 0 : 1;
		int mz = (vector_point.z - floor(vector_point.z)) < 0.5 ? 0 : 1;
		FLOAT c = FLOAT((mx + mz) % 2);
		return make_vector4<FLOAT>(c, c, c, 1);
	}
	}
	return make_vector4<FLOAT>(0, 0, 0, 0);
}

template <typename FLOAT>
DEVICE_PREFIX Vector4<FLOAT> RaycastBitangents(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	auto intersect = Intersect(pScene, origin, direction);
	if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
		return make_vector4<FLOAT>(0, 0, 0, 0);
	}
	Vector3<FLOAT> n = Intersect<FLOAT, IntersectTBN<FLOAT>>(origin, direction, intersect.Object).Bitangent;
	return make_vector4<FLOAT>((n.x + 1) / 2, (n.y + 1) / 2, (n.z + 1) / 2, 1);
}

template <typename FLOAT>
DEVICE_PREFIX Vector4<FLOAT> RaycastNormals(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	auto intersect = Intersect(pScene, origin, direction);
	if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
		return make_vector4<FLOAT>(0, 0, 0, 0);
	}
	Vector3<FLOAT> n = Intersect<FLOAT, IntersectNormal<FLOAT>>(origin, direction, intersect.Object).Normal;
	return make_vector4<FLOAT>((n.x + 1) / 2, (n.y + 1) / 2, (n.z + 1) / 2, 1);
}

template <typename FLOAT>
DEVICE_PREFIX Vector4<FLOAT> RaycastTangents(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	auto intersect = Intersect(pScene, origin, direction);
	if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
		return make_vector4<FLOAT>(0, 0, 0, 0);
	}
	Vector3<FLOAT> n = Intersect<FLOAT, IntersectTBN<FLOAT>>(origin, direction, intersect.Object).Tangent;
	return make_vector4<FLOAT>((n.x + 1) / 2, (n.y + 1) / 2, (n.z + 1) / 2, 1);
}

template <typename FLOAT, int RECURSE>
struct Raytrace
{
	DEVICE_PREFIX static Vector4<FLOAT> Do(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		// Start intersecting objects.
		auto intersect = Intersect(pScene, origin, direction);
		if (intersect.Lambda == FloatTraits<FLOAT>::Max) {
			return make_vector4<FLOAT>(0, 0, 0, 0);
		}
		Vector3<FLOAT> vector_point = origin + intersect.Lambda * direction;
		Vector3<FLOAT> vector_light = Normalize(make_vector3<FLOAT>(10, 10, -10) - vector_point);
		Vector3<FLOAT> vector_normal = Normalize(Intersect<FLOAT, IntersectNormal<FLOAT>>(origin, direction, intersect.Object).Normal);
		Vector4<FLOAT> color_diffuse = make_vector4<FLOAT>(1, 1, 1, 0);
		Vector4<FLOAT> color_specular = make_vector4<FLOAT>(1, 1, 1, 0);
		FLOAT scale_reflect = 0;
		FLOAT scale_refract = 0;
		FLOAT scale_ior = 1;
		switch (intersect.Object.Material) {
		case MATERIAL_COMMON:
		{
			if (intersect.Object.MaterialOffset != 0) {
				const MaterialCommon<FLOAT>* pMaterial = (const MaterialCommon<FLOAT>*)((const int*)&pScene + intersect.Object.MaterialOffset / 4);
				color_diffuse = pMaterial->Diffuse;
				color_specular = pMaterial->Specular;
				scale_reflect = pMaterial->Reflect.w;
				scale_refract = pMaterial->Refract.w;
				scale_ior = pMaterial->Ior;
			}
		}
		break;
		case MATERIAL_CHECKERBOARD_XZ:
		{
			int mx = (vector_point.x - floor(vector_point.x)) < 0.5 ? 0 : 1;
			int mz = (vector_point.z - floor(vector_point.z)) < 0.5 ? 0 : 1;
			FLOAT c = FLOAT((mx + mz) % 2);
			color_diffuse = make_vector4<FLOAT>(c, c, c, 0);
			color_specular = make_vector4<FLOAT>(1, 1, 1, 0);
			scale_reflect = 1;
			scale_refract = 0;
		}
		break;
		}
		Vector3<FLOAT> vector_incident = Normalize(direction);
		Vector3<FLOAT> vector_reflect = Reflect(vector_incident, vector_normal);
		Vector3<FLOAT> vector_refract = Refract(vector_incident, vector_normal, scale_ior);
		FLOAT scale_diffuse = Dot(vector_light, vector_normal);
		scale_diffuse = scale_diffuse > 0 ? scale_diffuse : 0;
		FLOAT scale_specular = Dot(vector_light, vector_reflect);
		scale_specular = scale_specular > 0 ? scale_specular : 0;
		scale_specular = Power(scale_specular, 100);
		if (RayShadow(pScene, vector_point + FLOAT(0.0001) * vector_normal, vector_light)) scale_diffuse *= FLOAT(0.5);
		Vector4<FLOAT> color = color_diffuse * scale_diffuse + color_specular * scale_specular;
		if (scale_refract > 0) {
			//Schlick's Approximation for reflectance.
			FLOAT R = SchlickApprox<FLOAT>(vector_incident, vector_normal, 1, scale_ior);
			scale_reflect = R;
			//Add refraction contribution.
			color = color + Raytrace<FLOAT, RECURSE - 1>::Do(pScene, vector_point + vector_refract * FLOAT(0.001), vector_refract) * scale_refract;
		}
		if (scale_reflect > 0) {
			color = color + Raytrace<FLOAT, RECURSE - 1>::Do(pScene, vector_point + vector_reflect * FLOAT(0.001), vector_reflect) * scale_reflect;
		}
		color.w = 1;
		return color;
	}
};

template <typename FLOAT>
struct Raytrace<FLOAT, -1>
{
	DEVICE_PREFIX static Vector4<FLOAT> Do(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		return make_vector4<FLOAT>(0, 0, 0, 0);
	}
};

template <typename FLOAT>
DEVICE_PREFIX void ComputeRay(const Matrix44<FLOAT>& inverse_mvp, FLOAT clipx, FLOAT clipy, Vector3<FLOAT>& origin, Vector3<FLOAT>& direction) DEVICE_SUFFIX {
	Vector4<FLOAT> v41 = Transform(inverse_mvp, make_vector4<FLOAT>(clipx, clipy, 0, 1));
	Vector4<FLOAT> v42 = Transform(inverse_mvp, make_vector4<FLOAT>(clipx, clipy, 1, 1));
	Vector3<FLOAT> ray_p1 = make_vector3<FLOAT>(v41.x / v41.w, v41.y / v41.w, v41.z / v41.w);
	Vector3<FLOAT> ray_p2 = make_vector3<FLOAT>(v42.x / v42.w, v42.y / v42.w, v42.z / v42.w);
	origin = ray_p1;
	direction = ray_p2 - ray_p1;
}

template <typename FLOAT>
DEVICE_PREFIX unsigned int MakePixel(const Vector4<FLOAT>& color) DEVICE_SUFFIX {
	unsigned char r = unsigned int(color.x < 0 ? 0 : (color.x > 1 ? 1 : color.x) * 255);
	unsigned char g = unsigned int(color.y < 0 ? 0 : (color.y > 1 ? 1 : color.y) * 255);
	unsigned char b = unsigned int(color.z < 0 ? 0 : (color.z > 1 ? 1 : color.z) * 255);
	unsigned char a = unsigned int(color.w < 0 ? 0 : (color.w > 1 ? 1 : color.w) * 255);
	return (a << 24) | (r << 16) | (g << 8) | (b << 0);
}

template<typename FLOAT>
struct SetPixel {
	DEVICE_PREFIX SetPixel(void* bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int x, int y) DEVICE_SUFFIX :
		bitmap_ptr(bitmap_ptr),
		bitmap_width(bitmap_width),
		bitmap_height(bitmap_height),
		bitmap_stride(bitmap_stride),
		x(x),
		y(y)
	{
	}
	DEVICE_PREFIX void Do(const Vector4<FLOAT>& color) const DEVICE_SUFFIX {
		void* pRaster = (unsigned char*)bitmap_ptr + bitmap_stride * y;
		unsigned int* pPixel = (unsigned int*)pRaster + x;
		*pPixel = MakePixel(color);
	}
	DEVICE_PREFIX int GetX() const DEVICE_SUFFIX { return x; }
	DEVICE_PREFIX int GetY() const DEVICE_SUFFIX { return y; }
	DEVICE_PREFIX int GetWidth() const DEVICE_SUFFIX { return bitmap_width; }
	DEVICE_PREFIX int GetHeight() const DEVICE_SUFFIX { return bitmap_height; }
	void* bitmap_ptr;
	int bitmap_width;
	int bitmap_height;
	int bitmap_stride;
	int x;
	int y;
};

template <typename FLOAT, typename COLORFUNC, typename SETPIXEL = SetPixel<FLOAT>>
DEVICE_PREFIX void ComputePixel(const Scene<FLOAT>& pScene, Matrix44<FLOAT> inverse_mvp, const SETPIXEL& setpixel, int superx = 1, int supery = 1) DEVICE_SUFFIX {
	if (setpixel.GetX() >= setpixel.GetWidth() || setpixel.GetY() >= setpixel.GetHeight()) return;
	Vector3<FLOAT> origin;
	Vector3<FLOAT> direction;
	Vector4<FLOAT> color = make_vector4<FLOAT>(0, 0, 0, 0);
	for (int y_supersample = 1; y_supersample <= supery; ++y_supersample) {
		for (int x_supersample = 1; x_supersample <= superx; ++x_supersample) {
			// Build a ray for this supersample.
			FLOAT vx = Lerp<FLOAT>(-1, +1, (setpixel.GetX() + FLOAT(x_supersample) / (superx + 1)) / setpixel.GetWidth());
			FLOAT vy = Lerp<FLOAT>(+1, -1, (setpixel.GetY() + FLOAT(y_supersample) / (supery + 1)) / setpixel.GetHeight());
			ComputeRay(inverse_mvp, vx, vy, origin, direction);
			// Compute intersection with plane.
			color = color + COLORFUNC::Do(pScene, origin, direction);
		}
	}
	setpixel.Do(color / FLOAT(superx * supery));
}

template<typename FLOAT>
struct DoRaycast {
public:
	DEVICE_PREFIX static Vector4<FLOAT> Do(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		return Raycast(pScene, origin, direction);
	}
};

template<typename FLOAT>
struct DoRaycastBitangents {
public:
	DEVICE_PREFIX static Vector4<FLOAT> Do(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		return RaycastBitangents(pScene, origin, direction);
	}
};

template<typename FLOAT>
struct DoRaycastNormals {
public:
	DEVICE_PREFIX static Vector4<FLOAT> Do(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		return RaycastNormals(pScene, origin, direction);
	}
};

template<typename FLOAT>
struct DoRaycastTangents {
public:
	DEVICE_PREFIX static Vector4<FLOAT> Do(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		return RaycastTangents(pScene, origin, direction);
	}
};

template<typename FLOAT, int RECURSION = 2>
struct DoRaytrace {
public:
	DEVICE_PREFIX static Vector4<FLOAT> Do(const Scene<FLOAT>& pScene, const Vector3<FLOAT>& origin, const Vector3<FLOAT>& direction) DEVICE_SUFFIX {
		return Raytrace<FLOAT, RECURSION>::Do(pScene, origin, direction);
	}
};