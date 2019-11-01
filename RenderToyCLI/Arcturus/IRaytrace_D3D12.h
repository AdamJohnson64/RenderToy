#pragma once

#include "IDevice3D_D3D12.h"
#include "IIndexBuffer_D3D12.h"
#include "IRenderTarget_D3D12.h"
#include "IVertexBuffer_D3D12.h"

namespace Arcturus
{
    void TestRaytracer(IDevice3D_D3D12* device, IRenderTarget_D3D12* renderTarget, IVertexBuffer_D3D12* vertexBuffer, IIndexBuffer_D3D12* indexBuffer);
}