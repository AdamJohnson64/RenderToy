#define _USE_MATH_DEFINES

#include "DrawingContextMesh.h"
#include "Vector.h"

#include <array>
#include <stdint.h>
#include <vector>

namespace Arcturus
{

    constexpr float SKIRTWIDTH = 0.5f;

    struct Rib
    {
        uint32_t Color;
        float Offset;
    };

    static float Max(float a, float b)
    {
        return a > b ? a : b;
    }

    static float Min(float a, float b)
    {
        return a < b ? a : b;
    }

    static float Saturate(float x)
    {
        return Max(0, Min(x, 1));
    }

    static uint32_t ToColorUint32(const Vec4& c)
    {
        uint32_t r = Saturate(c.X) * 255;
        uint32_t g = Saturate(c.Y) * 255;
        uint32_t b = Saturate(c.Z) * 255;
        uint32_t a = Saturate(c.W) * 255;
        return (a << 24) | (b << 16) | (g << 8) | (r << 0);
    }

    DrawingContextMesh::DrawingContextMesh()
    {
        reset();
    }

    void DrawingContextMesh::reset()
    {
        vertices.clear();
        indices.clear();
        _color = Vec4{ 1, 1, 1, 1 };
        _cursor = Vec2{ 0, 0 };
        _width = 1;
    }

    void DrawingContextMesh::setColor(const Vec4& color)
    {
        _color = color;
    }

    void DrawingContextMesh::setWidth(float width)
    {
        _width = width;
    }

    void DrawingContextMesh::moveTo(const Vec2& point)
    {
        _cursor = point;
    }

    void DrawingContextMesh::lineTo(const Vec2& point)
    {
        const float halfWidth = _width / 2;
        const Vec4 color2{ _color.X, _color.Y, _color.Z, 0 };
        const std::array<Rib, 4> ribs =
        {
            Rib { ToColorUint32(color2), -halfWidth - SKIRTWIDTH },
            Rib { ToColorUint32(_color), -halfWidth + SKIRTWIDTH },
            Rib { ToColorUint32(_color), halfWidth - SKIRTWIDTH },
            Rib { ToColorUint32(color2), halfWidth + SKIRTWIDTH },
        };
        Vec2 normal;
        {
            Vec2 delta{ point.X - _cursor.X, point.Y - _cursor.Y };
            float length = 1.0f / sqrtf(delta.X * delta.X + delta.Y * delta.Y);
            normal.X = -delta.Y * length;
            normal.Y = delta.X * length;
        }
        const uint32_t indexBase = static_cast<uint32_t>(vertices.size());
        for (const auto& rib : ribs)
        {
            vertices.push_back(Vertex{ Vec2 { _cursor.X + normal.X * rib.Offset, _cursor.Y + normal.Y * rib.Offset }, rib.Color });
        }
        _cursor = point;
        for (const auto& rib : ribs)
        {
            vertices.push_back(Vertex{ Vec2 { _cursor.X + normal.X * rib.Offset, _cursor.Y + normal.Y * rib.Offset }, rib.Color });
        }
        const uint32_t ribCount = static_cast<uint32_t>(ribs.size());
        for (uint32_t ribIndex = 0; ribIndex < ribCount - 1; ++ribIndex)
        {
            const uint32_t ribSpan1 = indexBase + ribIndex;
            const uint32_t ribSpan2 = ribSpan1 + ribCount;
            indices.push_back(ribSpan1 + 0);
            indices.push_back(ribSpan2 + 0);
            indices.push_back(ribSpan2 + 1);
            indices.push_back(ribSpan2 + 1);
            indices.push_back(ribSpan1 + 1);
            indices.push_back(ribSpan1 + 0);
        }
    }

    void DrawingContextMesh::drawCircle(const Vec2& point, float radius)
    {
        const uint32_t circumference = static_cast<uint32_t>(2 * M_PI * radius);
        const uint32_t spineCount = Max(4, Min(circumference / 4, 1000));
        const Vec4 color2{ _color.X, _color.Y, _color.Z, 0 };
        const float halfWidth = _width / 2;
        const std::array<Rib, 4> ribs =
        {
            Rib { ToColorUint32(color2), radius - halfWidth - SKIRTWIDTH },
            Rib { ToColorUint32(_color), radius - halfWidth + SKIRTWIDTH },
            Rib { ToColorUint32(_color), radius + halfWidth - SKIRTWIDTH },
            Rib { ToColorUint32(color2), radius + halfWidth + SKIRTWIDTH },
        };
        const uint32_t indexBase = static_cast<uint32_t>(vertices.size());
        for (int spineIndex = 0; spineIndex < spineCount; ++spineIndex)
        {
            const float angle = spineIndex * 2.0f * M_PI / spineCount;
            const Vec2 normal{ cosf(angle), sinf(angle) };
            for (const auto& rib : ribs)
            {
                vertices.push_back(Vertex{ Vec2 { point.X + normal.X * rib.Offset, point.Y + normal.Y * rib.Offset }, rib.Color });
            }
        }
        for (uint32_t spineIndex = 0; spineIndex < spineCount; ++spineIndex)
        {
            const uint32_t ribSpan1 = indexBase + spineIndex * ribs.size();
            const uint32_t ribSpan2 = indexBase + ((spineIndex + 1) % spineCount) * ribs.size();
            for (uint32_t ribIndex = 0; ribIndex < ribs.size() - 1; ++ribIndex)
            {
                indices.push_back(ribSpan1 + ribIndex + 0);
                indices.push_back(ribSpan2 + ribIndex + 0);
                indices.push_back(ribSpan2 + ribIndex + 1);
                indices.push_back(ribSpan2 + ribIndex + 1);
                indices.push_back(ribSpan1 + ribIndex + 1);
                indices.push_back(ribSpan1 + ribIndex + 0);
            }
        }
    }

    void DrawingContextMesh::drawRectangle(const Vec2& topLeft, const Vec2& bottomRight)
    {
        moveTo(topLeft);
        lineTo({ bottomRight.X, topLeft.Y });
        lineTo(bottomRight);
        lineTo({ topLeft.X, bottomRight.Y });
        lineTo(topLeft);
    }

    void DrawingContextMesh::fillCircle(const Vec2& point, float radius)
    {
        const uint32_t circumference = static_cast<uint32_t>(2 * M_PI * radius);
        const uint32_t spineCount = Max(4, Min(circumference / 4, 1000));
        const Vec4 color2{ _color.X, _color.Y, _color.Z, 0 };
        const std::array<Rib, 2> ribs =
        {
            Rib { ToColorUint32(_color), radius - SKIRTWIDTH },
            Rib { ToColorUint32(color2), radius + SKIRTWIDTH },
        };
        uint32_t indexBase = static_cast<uint32_t>(vertices.size());
        // Central point.
        vertices.push_back(Vertex{ point, ribs[0].Color });
        // Incremental rings.
        uint32_t ribBase = static_cast<uint32_t>(vertices.size());
        for (int spineIndex = 0; spineIndex < spineCount; ++spineIndex)
        {
            float angle = spineIndex * 2.0f * M_PI / spineCount;
            Vec2 normal{ cosf(angle), sinf(angle) };
            for (const auto& rib : ribs)
            {
                vertices.push_back(Vertex{ Vec2 { point.X + normal.X * rib.Offset, point.Y + normal.Y * rib.Offset }, rib.Color });
            }
        }
        // Central ring.
        for (uint32_t spineIndex = 0; spineIndex < spineCount; ++spineIndex)
        {
            indices.push_back(indexBase);
            indices.push_back(ribBase + spineIndex * ribs.size());
            indices.push_back(ribBase + ((spineIndex + 1) % spineCount) * ribs.size());
        }
        const uint32_t ribCount = ribs.size() - 1;
        for (uint32_t spineIndex = 0; spineIndex < spineCount; ++spineIndex)
        {
            const uint32_t ribSpan1 = ribBase + spineIndex * ribs.size();
            const uint32_t ribSpan2 = ribBase + ((spineIndex + 1) % spineCount) * ribs.size();
            for (uint32_t ribIndex = 0; ribIndex < ribCount; ++ribIndex)
            {
                indices.push_back(ribSpan1 + ribIndex + 0);
                indices.push_back(ribSpan2 + ribIndex + 0);
                indices.push_back(ribSpan2 + ribIndex + 1);
                indices.push_back(ribSpan2 + ribIndex + 1);
                indices.push_back(ribSpan1 + ribIndex + 1);
                indices.push_back(ribSpan1 + ribIndex + 0);
            }
        }
    }

    void DrawingContextMesh::fillRectangle(const Vec2& topLeft, const Vec2& bottomRight)
    {
        const uint32_t indexBase = vertices.size();
        const uint32_t color = ToColorUint32(_color);
        vertices.push_back(Vertex{ topLeft, color });
        vertices.push_back(Vertex{ Vec2 { bottomRight.X, topLeft.Y}, color });
        vertices.push_back(Vertex{ Vec2 { topLeft.X, bottomRight.Y}, color });
        vertices.push_back(Vertex{ bottomRight, color });
        indices.push_back(indexBase + 0);
        indices.push_back(indexBase + 1);
        indices.push_back(indexBase + 3);
        indices.push_back(indexBase + 3);
        indices.push_back(indexBase + 2);
        indices.push_back(indexBase + 0);
    }

    uint32_t DrawingContextMesh::vertexCount() const
    {
        return vertices.size();
    }

    const void* DrawingContextMesh::vertexPointer() const
    {
        return vertices.data();
    }

    uint32_t DrawingContextMesh::indexCount() const
    {
        return indices.size();
    }

    const uint32_t* DrawingContextMesh::indexPointer() const
    {
        return indices.data();
    }

}