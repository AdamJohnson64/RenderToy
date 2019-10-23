#pragma once

#include "MVector.h"

namespace Arcturus
{
    namespace Managed
    {
        public interface class IDrawingContext
        {
        public:
            virtual void setColor(Vec4 color) = 0;
            virtual void setWidth(float width) = 0;
            virtual void moveTo(Vec2 point) = 0;
            virtual void lineTo(Vec2 point) = 0;
            virtual void drawCircle(Vec2 point, float radius) = 0;
            virtual void drawRectangle(Vec2 topLeft, Vec2 bottomRight) = 0;
            virtual void fillCircle(Vec2 point, float radius) = 0;
            virtual void fillRectangle(Vec2 topLeft, Vec2 bottomRight) = 0;
        };
    }
}