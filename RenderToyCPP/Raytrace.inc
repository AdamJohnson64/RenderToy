////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Device Code.
////////////////////////////////////////////////////////////////////////////////

// Basic Math Primitives.
DEVICE_PROTO double3 operator-(const double3& val) { return make_double3(-val.x, -val.y, -val.z); }
DEVICE_PROTO double4 operator-(const double4& val) { return make_double4(-val.z, -val.y, -val.z, -val.w); }
DEVICE_PROTO double3 operator+(const double3& lhs, const double3& rhs) { return make_double3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
DEVICE_PROTO double4 operator+(const double4& lhs, const double4& rhs) { return make_double4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
DEVICE_PROTO double3 operator-(const double3& lhs, const double3& rhs) { return make_double3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
DEVICE_PROTO double4 operator-(const double4& lhs, const double4& rhs) { return make_double4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
DEVICE_PROTO double3 operator*(const double3& lhs, double rhs) { return make_double3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs); }
DEVICE_PROTO double3 operator*(double lhs, const double3& rhs) { return rhs * lhs; }
DEVICE_PROTO double4 operator*(const double4& lhs, double rhs) { return make_double4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs); }
DEVICE_PROTO double4 operator*(double lhs, const double4& rhs) { return rhs * lhs; }
DEVICE_PROTO double3 operator/(const double3 &lhs, double rhs) { return lhs * (1 / rhs); }
DEVICE_PROTO double4 operator/(const double4 &lhs, double rhs) { return lhs * (1 / rhs); }

// Common Math Primitives.
DEVICE_PROTO double Clamp(double min, double max, double val) { return val < min ? min : (val > max ? max : val); }
DEVICE_PROTO double Dot(const double3& lhs, const double3& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
DEVICE_PROTO double Length(const double3& val) { return norm3d(val.x, val.y, val.z); }
DEVICE_PROTO double Lerp(double y1, double y2, double x) { return y1 + (y2 - y1) * x; }
DEVICE_PROTO double3 Normalize(const double3 &val) { return val * rnorm3d(val.x, val.y, val.z); }

// Matrix Math.
DEVICE_PROTO double3 TransformPoint(const Matrix4D& m, const double3& p) {
	return make_double3(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z + m.M[12],
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z + m.M[13],
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z + m.M[14]);
}

DEVICE_PROTO double3 TransformVector(const Matrix4D& m, const double3& p) {
	return make_double3(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z,
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z,
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z);
}

DEVICE_PROTO double4 Transform(const Matrix4D& m, const double4& p) {
	return make_double4(
		m.M[0] * p.x + m.M[4] * p.y + m.M[8] * p.z + m.M[12] * p.w,
		m.M[1] * p.x + m.M[5] * p.y + m.M[9] * p.z + m.M[13] * p.w,
		m.M[2] * p.x + m.M[6] * p.y + m.M[10] * p.z + m.M[14] * p.w,
		m.M[3] * p.x + m.M[7] * p.y + m.M[11] * p.z + m.M[15] * p.w);
}

DEVICE_PROTO double3 Reflect(const double3& incident, const double3& normal) {
	return incident - 2 * Dot(incident, normal) * normal;
}

DEVICE_PROTO double3 Refract(const double3& incident, const double3& normal, double ior) {
	double cosi = Clamp(-1, 1, Dot(incident, normal));
	double etai = 1, etat = ior;
	double3 n = normal;
	if (cosi < 0) { cosi = -cosi; }
	else { double tmp = etai; etai = etat; etat = tmp; n = -normal; }
	double eta = etai / etat;
	double k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? make_double3(0, 0, 0) : (eta * incident + (eta * cosi - sqrt(k)) * n);
}

DEVICE_PROTO double SchlickApprox(const double3& incident, const double3 &normal, double ior1, double ior2) {
	double coeff = (ior1 - ior2) / (ior1 + ior2);
	coeff = coeff * coeff;
	return coeff + (1 - coeff) * pow((1 - Dot(-incident, normal)), 5);
}

// Geometric Math.
DEVICE_PROTO double IntersectPlane(const double3 &origin, const double3 &direction, const double3 &plane_normal, double plane_distance) {
	return (plane_distance - Dot(plane_normal, origin)) / Dot(plane_normal, direction);
}

template <typename T> DEVICE_PROTO void SetLambda(T& result, double value) { result.Lambda = value; }
DEVICE_PROTO void SetNormal(IntersectNormal& result, double3 value) { result.Normal = value; }
DEVICE_PROTO void SetNormal(IntersectSimple& result, double3 value) {}
DEVICE_PROTO double3 GetNormal(const IntersectSimple& result) { return make_double3(0, 0, 0); }
DEVICE_PROTO double3 GetNormal(const IntersectNormal& result) { return result.Normal; }

template <typename T>
DEVICE_PROTO T IntersectCube(const double3 &origin, const double3 &direction) {
	T result;
	double3 face_normal[] = {
		make_double3(-1,0,0), make_double3(+1,0,0),
		make_double3(0,-1,0), make_double3(0,+1,0),
		make_double3(0,0,-1), make_double3(0,0,+1),
	};
	double best_lambda = DBL_MAX;
	for (int face_index = 0; face_index < 6; ++face_index) {
		double lambda = IntersectPlane(origin, direction, face_normal[face_index], 1);
		if (lambda < 0 || lambda > best_lambda) continue;
		double3 point = origin + lambda * direction;
		// Check that the point is inside every other plane.
		bool use_face = true;
		for (int check_face = 0; check_face < 6; ++check_face) {
			if (face_index == check_face) continue;
			double inside = Dot(point, face_normal[check_face]) - 1;
			if (inside > 0) {
				use_face = false;
				break;
			}
		}
		if (use_face) {
			best_lambda = lambda;
			SetNormal(result, face_normal[face_index]);
		}
	}
	SetLambda(result, best_lambda);
	return result;
}

template <typename T>
DEVICE_PROTO T IntersectPlane(const double3 &origin, const double3 &direction) {
	T result;
	const double PLANE_DISTANCE = 0;
	const double3 PLANE_NORMAL = make_double3(0, 1, 0);
	SetLambda(result, (PLANE_DISTANCE - Dot(PLANE_NORMAL, origin)) / Dot(PLANE_NORMAL, direction));
	SetNormal(result, PLANE_NORMAL);
	return result;
}

template <typename T>
DEVICE_PROTO T IntersectSphere(const double3 &origin, const double3 &direction) {
	T result;
	const double SPHERE_RADIUS = 1;
	double a = Dot(direction, direction);
	double b = 2 * Dot(origin, direction);
	double c = Dot(origin, origin) - SPHERE_RADIUS * SPHERE_RADIUS;
	// If the determinant is negative then there are no real roots and this will be NaN.
	double det = sqrt(b * b - 4 * a * c);
	// "a" cannot be negative so (worst case) these lambdas are +Inf.
	double den = 2 * a;
	double lambda1 = (-b - det) / den;
	double lambda2 = (-b + det) / den;
	double lambda_best = DBL_MAX;
	if (lambda1 >= 0 && lambda1 < lambda_best) lambda_best = lambda1;
	if (lambda2 >= 0 && lambda2 < lambda_best) lambda_best = lambda2;
	SetLambda(result, lambda_best);
	SetNormal(result, origin + lambda_best * direction);
	return result;
}

template <typename T>
DEVICE_PROTO T Intersect(const double3 &origin, const double3 &direction, GeometryType geometry) {
	switch (geometry) {
	case GEOMETRY_PLANE:
		return IntersectPlane<T>(origin, direction);
	case GEOMETRY_SPHERE:
		return IntersectSphere<T>(origin, direction);
	case GEOMETRY_CUBE:
		return IntersectCube<T>(origin, direction);
	default:
		T result;
		SetLambda(result, DBL_MAX);
		return result;
	}
}

template <typename T>
DEVICE_PROTO T Intersect(const double3 &origin, const double3 &direction, const SceneObject &scene_object) {
	double3 transformed_origin = TransformPoint(scene_object.TransformInverse, origin);
	double3 transformed_direction = TransformVector(scene_object.TransformInverse, direction);
	T result = Intersect<T>(transformed_origin, transformed_direction, scene_object.Geometry);
	SetNormal(result, TransformVector(scene_object.Transform, GetNormal(result)));
	return result;
}

// Raytracer Ray Tests.
DEVICE_PROTO bool RayShadow(const Scene *pScene, const double3 &origin, const double3 &direction) {
	for (int i = 0; i < pScene->ObjectCount; ++i) {
		const SceneObject &scene_object = pScene->Objects[i];
		double3 transformed_origin = TransformPoint(scene_object.TransformInverse, origin);
		double3 transformed_direction = TransformVector(scene_object.TransformInverse, direction);
		double lambda = Intersect<IntersectSimple>(transformed_origin, transformed_direction, scene_object.Geometry).Lambda;
		if (lambda >= 0 && lambda < DBL_MAX) return true;
	}
	return false;
}

DEVICE_PROTO double4 RayCast(const Scene* pScene, const double3& origin, const double3& direction) {
	double best_lambda = DBL_MAX;
	const SceneObject *best_object = nullptr;
	for (int i = 0; i < pScene->ObjectCount; ++i) {
		const SceneObject &scene_object = pScene->Objects[i];
		double lambda = Intersect<IntersectSimple>(origin, direction, scene_object).Lambda;
		if (lambda >= 0 && lambda < best_lambda) {
			best_lambda = lambda;
			best_object = &scene_object;
		}
	}
	if (best_lambda == DBL_MAX) {
		return make_double4(0, 0, 0, 0);
	}
	double3 vector_point = origin + best_lambda * direction;
	switch (best_object->Material) {
	case MATERIAL_COMMON:
	{
		if (best_object->MaterialOffset != 0) {
			MaterialCommon *pMaterial = (MaterialCommon*)((unsigned char*)pScene + best_object->MaterialOffset);
			return pMaterial->Diffuse;
		}
	}
	break;
	case MATERIAL_CHECKERBOARD_XZ:
	{
		int mx = (vector_point.x - floor(vector_point.x)) < 0.5 ? 0 : 1;
		int mz = (vector_point.z - floor(vector_point.z)) < 0.5 ? 0 : 1;
		double c = (mx + mz) % 2;
		return make_double4(c, c, c, 1);
	}
	}
	return make_double4(0, 0, 0, 0);
}

DEVICE_PROTO double4 RayCastNormals(const Scene* pScene, const double3& origin, const double3& direction) {
	double best_lambda = DBL_MAX;
	const SceneObject *best_object = nullptr;
	for (int i = 0; i < pScene->ObjectCount; ++i) {
		const SceneObject &scene_object = pScene->Objects[i];
		double lambda = Intersect<IntersectSimple>(origin, direction, scene_object).Lambda;
		if (lambda >= 0 && lambda < best_lambda) {
			best_lambda = lambda;
			best_object = &scene_object;
		}
	}
	if (best_lambda == DBL_MAX) {
		return make_double4(0, 0, 0, 0);
	}
	double3 n = Intersect<IntersectNormal>(origin, direction, *best_object).Normal;
	return make_double4((n.x + 1) / 2, (n.y + 1) / 2, (n.z + 1) / 2, 1);
}

template <int RECURSE>
DEVICE_PROTO double4 RayColor(const Scene *pScene, const double3 &origin, const double3 &direction) {
	// Start intersecting objects.
	double best_lambda = DBL_MAX;
	const SceneObject *best_object = nullptr;
	for (int i = 0; i < pScene->ObjectCount; ++i) {
		const SceneObject &scene_object = pScene->Objects[i];
		double lambda = Intersect<IntersectSimple>(origin, direction, scene_object).Lambda;
		if (lambda >= 0 && lambda < best_lambda) {
			best_lambda = lambda;
			best_object = &scene_object;
		}
	}
	if (best_lambda == DBL_MAX) {
		return make_double4(0, 0, 0, 0);
	}
	double3 vector_point = origin + best_lambda * direction;
	double3 vector_light = Normalize(make_double3(10, 10, -10) - vector_point);
	double3 vector_normal = Normalize(Intersect<IntersectNormal>(origin, direction, *best_object).Normal);
	double4 color_diffuse = make_double4(1, 1, 1, 0);
	double4 color_specular = make_double4(1, 1, 1, 0);
	double scale_reflect = 0;
	double scale_refract = 0;
	double scale_ior = 1;
	switch (best_object->Material) {
	case MATERIAL_COMMON:
	{
		if (best_object->MaterialOffset != 0) {
			MaterialCommon *pMaterial = (MaterialCommon*)((unsigned char*)pScene + best_object->MaterialOffset);
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
		double c = (mx + mz) % 2;
		color_diffuse = make_double4(c, c, c, 0);
		color_specular = make_double4(1, 1, 1, 0);
		scale_reflect = 1;
		scale_refract = 0;
	}
	break;
	}
	double3 vector_incident = Normalize(direction);
	double3 vector_reflect = Reflect(vector_incident, vector_normal);
	double3 vector_refract = Refract(vector_incident, vector_normal, scale_ior);
	double scale_diffuse = Dot(vector_light, vector_normal);
	scale_diffuse = scale_diffuse > 0 ? scale_diffuse : 0;
	double scale_specular = Dot(vector_light, vector_reflect);
	scale_specular = scale_specular > 0 ? scale_specular : 0;
	scale_specular = pow(scale_specular, 100);
	if (RayShadow(pScene, vector_point + 0.0001 * vector_normal, vector_light)) scale_diffuse *= 0.5;
	double4 color = color_diffuse * scale_diffuse + color_specular * scale_specular;
	if (scale_refract > 0) {
		//Schlick's Approximation for reflectance.
		double R = SchlickApprox(vector_incident, vector_normal, 1, scale_ior);
		scale_reflect = R;
		//Add refraction contribution.
		color = color + RayColor<RECURSE - 1>(pScene, vector_point + vector_refract * 0.001, vector_refract) * scale_refract;
	}
	if (scale_reflect > 0) {
		color = color + RayColor<RECURSE - 1>(pScene, vector_point + vector_reflect * 0.001, vector_reflect) * scale_reflect;
	}
	color.w = 1;
	return color;
}

template <>
DEVICE_PROTO double4 RayColor<-1>(const Scene *pScene, const double3 &origin, const double3 &direction) {
	return make_double4(0, 0, 0, 0);
}

struct DoRayCast {
public:
	DEVICE_PROTO static double4 Do(const Scene* pScene, const double3& origin, const double3& direction) {
		return RayCast(pScene, origin, direction);
	}
};

struct DoRayCastNormals {
public:
	DEVICE_PROTO static double4 Do(const Scene* pScene, const double3& origin, const double3& direction) {
		return RayCastNormals(pScene, origin, direction);
	}
};

struct DoRayColor {
public:
	DEVICE_PROTO static double4 Do(const Scene* pScene, const double3& origin, const double3& direction) {
		return RayColor<2>(pScene, origin, direction);
	}
};

DEVICE_PROTO void ComputeRay(const Matrix4D &inverse_mvp, double clipx, double clipy, double3 &origin, double3 &direction) {
	double4 v41 = Transform(inverse_mvp, make_double4(clipx, clipy, 0, 1));
	double4 v42 = Transform(inverse_mvp, make_double4(clipx, clipy, 1, 1));
	double3 ray_p1 = make_double3(v41.x / v41.w, v41.y / v41.w, v41.z / v41.w);
	double3 ray_p2 = make_double3(v42.x / v42.w, v42.y / v42.w, v42.z / v42.w);
	origin = ray_p1;
	direction = ray_p2 - ray_p1;
}

DEVICE_PROTO unsigned int MakePixel(const double4 &color) {
	unsigned char r = color.x < 0 ? 0 : (color.x > 1 ? 1 : color.x) * 255;
	unsigned char g = color.y < 0 ? 0 : (color.y > 1 ? 1 : color.y) * 255;
	unsigned char b = color.z < 0 ? 0 : (color.z > 1 ? 1 : color.z) * 255;
	unsigned char a = color.w < 0 ? 0 : (color.w > 1 ? 1 : color.w) * 255;
	return (a << 24) | (r << 16) | (g << 8) | (b << 0);
}

template <typename T, int X_SUPERSAMPLES, int Y_SUPERSAMPLES>
DEVICE_PROTO void cudaFill2(const Scene *pScene, Matrix4D inverse_mvp, void *bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride, int x, int y) {
	if (x >= bitmap_width || y >= bitmap_height) return;
	// Generate untransformed ray.
	double3 origin;
	double3 direction;
	double4 color = make_double4(0, 0, 0, 0);
	for (int y_supersample = 1; y_supersample <= Y_SUPERSAMPLES; ++y_supersample) {
		for (int x_supersample = 1; x_supersample <= X_SUPERSAMPLES; ++x_supersample) {
			// Build a ray for this supersample.
			double vx = Lerp(-1, +1, (x + x_supersample / (X_SUPERSAMPLES + 1.0)) / bitmap_width);
			double vy = Lerp(+1, -1, (y + y_supersample / (Y_SUPERSAMPLES + 1.0)) / bitmap_height);
			ComputeRay(inverse_mvp, vx, vy, origin, direction);
			// Compute intersection with plane.
			color = color + T::Do(pScene, origin, direction);
		}
	}
	color = color / (X_SUPERSAMPLES * Y_SUPERSAMPLES);
	// Fill in the pixel.
	void *pRaster = (unsigned char*)bitmap_ptr + bitmap_stride * y;
	void *pPixel = (unsigned char*)pRaster + 4 * x;
	*(unsigned int*)pPixel = MakePixel(color);
}