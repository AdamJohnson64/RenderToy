#include "DrawingContextMesh.h"
#include "MDevice3D.h"
#include "MIDrawingContext.h"
#include "MStub.h"
#include "Vector.h"

#include <stdint.h>

namespace Arcturus
{
    namespace Managed
    {
        public ref class DrawingContextMesh : public IDrawingContext
        {
        public:
            DrawingContextMesh()
            {
                thunk = new Arcturus::DrawingContextMesh();
            }
            ~DrawingContextMesh()
            {
                if (thunk != nullptr)
                {
                    delete thunk;
                    thunk = nullptr;
                }
            }
            !DrawingContextMesh()
            {
                if (thunk != nullptr)
                {
                    delete thunk;
                    thunk = nullptr;
                }
            }
            void reset()
            {
                thunk->reset();
            }
            virtual void setColor(Vec4 color)
            {
                thunk->setColor(Arcturus::Vec4{ static_cast<float>(color.X), static_cast<float>(color.Y), static_cast<float>(color.Z), static_cast<float>(color.W) });
            }
            virtual void setWidth(float width)
            {
                thunk->setWidth(width);
            }
            virtual void moveTo(Vec2 point)
            {
                thunk->moveTo(Arcturus::Vec2{ static_cast<float>(point.X), static_cast<float>(point.Y) });
            }
            virtual void lineTo(Vec2 point)
            {
                thunk->lineTo(Arcturus::Vec2{ static_cast<float>(point.X), static_cast<float>(point.Y) });
            }
            virtual void drawCircle(Vec2 point, float radius)
            {
                thunk->drawCircle(Arcturus::Vec2{ static_cast<float>(point.X), static_cast<float>(point.Y) }, radius);
            }
            virtual void drawRectangle(Vec2 topLeft, Vec2 bottomRight)
            {
                thunk->drawRectangle(
                    Arcturus::Vec2{ static_cast<float>(topLeft.X), static_cast<float>(topLeft.Y) },
                    Arcturus::Vec2{ static_cast<float>(bottomRight.X), static_cast<float>(bottomRight.Y) }
                );
            }
            virtual void fillCircle(Vec2 point, float radius)
            {
                thunk->fillCircle(Arcturus::Vec2{ static_cast<float>(point.X), static_cast<float>(point.Y) }, radius);
            }
            virtual void fillRectangle(Vec2 topLeft, Vec2 bottomRight)
            {
                thunk->fillRectangle(
                    Arcturus::Vec2{ static_cast<float>(topLeft.X), static_cast<float>(topLeft.Y) },
                    Arcturus::Vec2{ static_cast<float>(bottomRight.X), static_cast<float>(bottomRight.Y) }
                );
            }
            uint32_t vertexCount()
            {
                return thunk->vertexCount();
            }
            System::IntPtr vertexPointer()
            {
                return System::IntPtr(const_cast<void*>(thunk->vertexPointer()));
            }
            uint32_t indexCount()
            {
                return thunk->indexCount();
            }
            System::IntPtr indexPointer()
            {
                return System::IntPtr(const_cast<uint32_t*>(thunk->indexPointer()));
            }
        private:
            Arcturus::DrawingContextMesh* thunk;
        };
    }
}