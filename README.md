# RenderToy

## Inspiration
I've been working on Linux for too long.

## Goals
3D Graphics is a huge topic. Back when I started studying this in the 90s
some of the concepts were as immature as my view of them. My first
rasterizer was in screen space and interpolated incorrectly (without the
homogeneous divide) and I didn't even know what linear algebra was.
These were clearly good times and bad code - awesome.

So I'm going to do a refresh and try to build this sample project mostly
as a teaching aid for people that want to learn about 3D and how it works
under the hood. A key feature is that everything I touch on will have a
software reference implementation so you can see the math behind the
hardware renderer. I also want to look at non-realtime rendering by taking
a walk through Raytrace Lane.

Software goals are:
- Simplicity.
  - Everything simple, brief and self-explanatory.
  - Performance is always good but it's not our concern here.
- Flexibility.
  - I'm not building an OpenGL state machine; we want to drop in concepts
    such that we can tune out things we don't want to think about.
  - Deal with each concept in isolation so we can demonstrate it.
- Reusability.
  - Parts of this can be a tooling platform for trying other things.
  - Split out controls (etc) for the express purpose of simple reuse.

I'm not touching on OpenCL, DirectCompute or CUDA, but I won't rule it out.

## Ideas
Things we could cover:

- Basic mathematical concepts.
  - 4D Vectors.
    - Basic representation.
	- The Dot Product.
	- The Cross Product.
    - Homogeneous Coordinates.
	- Advanced: X86 optimizations (SOA, AOS and SIMD).
  - 4D Matrices & Linear Algebra.
    - Applying the math (fundamentals).
    - Translations.
	- Scaling.
	- Rotation.
	- Composition.
  - Quaternions.
	- Euler angles and gimbal lock (the "why").
	- Applying the math (fundamentals).
- The Scene Graph.
  - Triangles & Meshes.
  - Object Space & Transform hierarchies.
  - Traversal and rendering.
  - State based optimization and batching.
- A Simple Rendering Pipeline ("My First Renderer(TM)").
  - Initially wireframe only (to highlight geometry/mesh principles).
  - Model, View, Projection Transforms.
  - Homogeneous Coordinates & The Viewport.
  - The W-Divide (1/w) & Perspective.
  - Homogeneous Clipping.
- Parametric Surfaces.
  - Bezier Splines ("How To Build A Boat(TM)")
  - Bezier Patches and Smooth Surfaces.
  - Relation to Tesselator Shaders.
- DirectX 9 Hardware Rendering Pipelines.
  - Vertex Shaders.
  - Pixel Shaders.
- Rendering APIs.
  - Pure software from the ground up.
  - OpenGL; why you shouldn't use it anymore.
  - OpenGL ES; very relevant for nearly all development.
  - Vulkan; watch this one carefully, it's the next big thing.
  - DirectX; when to use it (XBox), and when not to use it (everything else).
- Classic Whitted Raytracing.
  - The Eye Ray & Screen.
  - Implicit Surfaces.
  - Representing Rays (the photon analog).
  - Solving Raytrace Equations.
    - Ray/Plane.
	- Ray/Sphere.
  - A simple demo under WPF control.