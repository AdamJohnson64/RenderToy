#include <cmath>
#include <functional>
#include <limits>

#include <windows.h>

struct Vector3 { float x, y, z; };
struct Vector4 { float x, y, z, w; };

Vector3 operator*(const Vector3&a, float b) { return Vector3{ a.x *b, a.y * b, a.z * b }; }
Vector4 operator*(const Vector4&a, float b) { return Vector4{ a.x *b, a.y * b, a.z * b, a.w * b }; }
Vector3 Cross(const Vector3& a, const Vector3& b) { return Vector3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
float   Dot(const Vector3& a, const Vector3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float   Dot(const Vector4& a, const Vector4 &b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
float   LengthSquared(const Vector3& a) { return Dot(a, a); }
float   LengthSquared(const Vector4& a) { return Dot(a, a); }
float   Length(const Vector3& a) { return sqrt(LengthSquared(a)); }
float   Length(const Vector4& a) { return sqrt(LengthSquared(a)); }
Vector3 Normalize(const Vector3& a) { return a * (1.0f / LengthSquared(a)); }
Vector4 Normalize(const Vector4& a) { return a * (1.0f / LengthSquared(a)); }

struct Plane { Vector3 n; float d; }; 
struct Ray { Vector3 o, d; };
struct Sphere { float r; };

bool Intersect(const Ray& r, const Plane& p, float& lambda) {
	float det = Dot(p.n, r.d);
	if (det == 0) return false;
	lambda = (p.d - Dot(p.n, r.o)) / det;
	return true;
}

bool Intersect(const Ray& r, const Sphere& o, float& lambda1, float& lambda2) {
	float a = Dot(r.d, r.d);
	float b = 2 * Dot(r.o, r.d);
	float c = Dot(r.o, r.o) - o.r * o.r;
	float det = b * b - 4 * a * c;
	if (det <= 0) return false;
	det = sqrt(det);
	float den = 2 * a;
	lambda1 = (-b - det) / den;
	lambda2 = (-b + det) / den;
	return true;
}

int main() {
	Plane plane{ Normalize(Vector3{0.0f, 1.0f, 0.0f}), 0.0f };
	Sphere sphere{ 1.0f };
	static const int SCREEN_WIDTH = 512;
	static const int SCREEN_HEIGHT = 512;
	static Vector3 pixels[SCREEN_WIDTH * SCREEN_HEIGHT];
	for (int y = 0; y < SCREEN_HEIGHT; ++y) {
		for (int x = 0; x < SCREEN_WIDTH; ++x) {
			Ray ray{
				Vector3{0.0f, 0.5f, -1.5f},
				Normalize(Vector3{x - 0.5f * SCREEN_WIDTH, 0.5f * SCREEN_HEIGHT - y, 100.0f})
			};
			Vector3 output_color = Vector3{ 0, 0, 0 };
			float lambda_best = std::numeric_limits<float>::infinity();
			// Intersect with the plane.
			{
				float lambda = 0.0f;
				bool hit_plane = Intersect(ray, plane, lambda);
				if (hit_plane && lambda >= 0.0f && lambda < lambda_best) {
					lambda_best = lambda;
					output_color = Vector3 { 0.8f, 0.8f, 0.8f };
				}
			}
			// Intersect with the sphere.
			{
				float lambda1 = 0.0f;
				float lambda2 = 0.0f;
				bool hit_sphere = Intersect(ray, sphere, lambda1, lambda2);
				if (hit_sphere) {
					if (lambda1 >= 0.0f && lambda1 < lambda_best) {
						lambda_best = lambda1;
						output_color = Vector3{ 1, 0, 0 };
					}
					if (lambda2 >= 0.0f && lambda2 < lambda_best) {
						lambda_best = lambda2;
						output_color = Vector3{ 1, 0, 0 };
					}
				}
			}
			// Draw the result.
			pixels[x + y * SCREEN_WIDTH] = output_color;
		}
	}
	auto lam = [](HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam) -> LRESULT {
		if (Msg == WM_CLOSE) {
			DestroyWindow(hWnd);
			return 0;
		}
		if (Msg == WM_PAINT) {
			PAINTSTRUCT ps = { 0 };
			BeginPaint(hWnd, &ps);
			HDC backbuffer_hdc = CreateCompatibleDC(ps.hdc);
			HBITMAP backbuffer_hbitmap = CreateCompatibleBitmap(ps.hdc, SCREEN_WIDTH, SCREEN_HEIGHT);
			HBITMAP backbuffer_oldhbitmap = (HBITMAP)SelectObject(backbuffer_hdc, backbuffer_hbitmap);
			for (int y = 0; y < SCREEN_HEIGHT; ++y) {
				for (int x = 0; x < SCREEN_WIDTH; ++x) {
					const Vector3& pixel = pixels[x + y * SCREEN_WIDTH];
					int r = min(max(0, pixel.x * 255), 255);
					int g = min(max(0, pixel.y * 255), 255);
					int b = min(max(0, pixel.z * 255), 255);
					SetPixel(backbuffer_hdc, x, y, RGB(r, g, b));
				}
			}
			BitBlt(ps.hdc, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, backbuffer_hdc, 0, 0, SRCCOPY);
			SelectObject(backbuffer_hdc, backbuffer_oldhbitmap);
			DeleteObject(backbuffer_hbitmap);
			DeleteObject(backbuffer_hdc);
			EndPaint(hWnd, &ps);
		}
		return DefWindowProc(hWnd, Msg, wParam, lParam);
	};
	WNDCLASS wc = { 0 };
	wc.lpszClassName = "raytracer_class";
	wc.lpfnWndProc = lam;
	RegisterClass(&wc);
	RECT rect{ 64, 64, 64 + SCREEN_WIDTH, 64 + SCREEN_HEIGHT };
	AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
	HWND hwnd = CreateWindow("raytracer_class", "raytracer", WS_OVERLAPPEDWINDOW, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, nullptr, nullptr);
	ShowWindow(hwnd, SW_SHOW);
	MSG msg;
	while (GetMessage(&msg, hwnd, 0, 0) == TRUE) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return 0;
}