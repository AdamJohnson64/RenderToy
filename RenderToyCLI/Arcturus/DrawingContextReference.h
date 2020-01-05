#pragma once

#include "IDrawingContext.h"
#include "Vector.h"

#include <vector>

namespace Arcturus
{
    class DrawingContextReference : public IDrawingContext
    {
    public:
        DrawingContextReference();
        void reset();
        void renderTo(void* pixels, uint32_t width, uint32_t height, uint32_t stride);
        // IDrawingContext Implementation.
        void setColor(const Vec4& color) override;
        void setWidth(float width) override;
        void moveTo(const Vec2& point) override;
        void lineTo(const Vec2& point) override;
        void drawCircle(const Vec2& point, float radius) override;
        void drawRectangle(const Vec2& topLeft, const Vec2& bottomRight) override;
        void fillCircle(const Vec2& point, float radius) override;
        void fillRectangle(const Vec2& topLeft, const Vec2& bottomRight) override;
    private:
        Vec4 _color;
        Vec2 _cursor;
        float _width;
        std::vector<uint8_t> _data;
    };
}