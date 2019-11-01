#pragma once

#include "Vector.h"

namespace Arcturus
{
    class IDrawingContext
    {
    public:
        virtual void setColor(const Vec4& color) = 0;
        virtual void setWidth(float width) = 0;
        virtual void moveTo(const Vec2& point) = 0;
        virtual void lineTo(const Vec2& point) = 0;
        virtual void drawCircle(const Vec2& point, float radius) = 0;
        virtual void drawRectangle(const Vec2& topLeft, const Vec2& bottomRight) = 0;
        virtual void fillCircle(const Vec2& point, float radius) = 0;
        virtual void fillRectangle(const Vec2& topLeft, const Vec2& bottomRight) = 0;
    };
}