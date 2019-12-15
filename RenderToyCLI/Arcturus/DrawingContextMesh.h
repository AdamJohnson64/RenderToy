#pragma once

#include "IDrawingContext.h"
#include "IDrawingContextAccess.h"
#include "Vector.h"

#include <vector>

namespace Arcturus
{
    class DrawingContextMesh : public IDrawingContext, public IDrawingContextAccess
    {
    public:
        DrawingContextMesh();
        void reset();
        // IDrawingContext Implementation.
        void setColor(const Vec4& color) override;
        void setWidth(float width) override;
        void moveTo(const Vec2& point) override;
        void lineTo(const Vec2& point) override;
        void drawCircle(const Vec2& point, float radius) override;
        void drawRectangle(const Vec2& topLeft, const Vec2& bottomRight) override;
        void fillCircle(const Vec2& point, float radius) override;
        void fillRectangle(const Vec2& topLeft, const Vec2& bottomRight) override;
        // IDrawingContextAccess Implementation.
        uint32_t vertexCount() const override;
        const void* vertexPointer() const override;
        uint32_t indexCount() const override;
        const uint32_t* indexPointer() const override;
    private:
        Vec4 _color;
        Vec2 _cursor;
        float _width;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
    };
}