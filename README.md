# RenderToy

## Latest
You'll see references to Arcturus starting to appear in the code. Arcturus is
a highly meaningful name because it's a big thing, and this thing will be big.
Ahem. That's...uh...really all that means (so we have a namespace).

One thing we all end up doing at some point is using a graphics abstraction to
handle things like D3D12 vs Vulkan so the goal here is to build such an
abstraction. CUDA is also likely to go away since in the time this application
was written things like DX Raytracing didn't exist. We now have DX Raytracing
(DXR) so we'll want to use that. The educational aspect of this application
remains valid but with a change of secondary focus toward realtime rendering.

OpenVR will remain a big feature in here since I have to justify spending
money on a Vive Pro, and also because it's fun. We'll probably have a play
with the ideas surrounding VR usage and effective VR UI analogues.

UWP goes out of the window because it imposes far too many restrictions on what
code we can leverage (it doesn't even allow CUDA...seriously WTF). If the
raytracer ends up as DXR then we'll get that via D3D12 natively.

AMP goes out of the window because it sucks mightily and the compiler is slow
as the pitch-drop experiment in slow-motion. There's also no future for AMP
beyond its ability to raytrace in UWP which we expect to be taken up by DXR.

The war of the GPU C++ API is far from over. We're going to see many more
DirectComputes, AMPs, and CUDAs. Thankfully we never got to OpenCL but that's
okay because that platform is history now.

## Inspiration
I've been working on Linux for too long.

## Goals
3D Graphics is a huge topic. Back when I started studying this in the 90s some
of the concepts were as immature as my view of them. My first rasterizer was
in screen space and interpolated incorrectly (without the homogeneous divide)
and I didn't even know what linear algebra was.

This project is a refresh as I build a sample project mostly as a teaching aid
for people that want to learn about 3D and how it works under the hood. A key
feature is that everything I touch on will have a software reference
implementation so you can see the math behind the hardware renderer. I also
want to look at non-realtime rendering by taking a walk through Raytrace Lane.

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

## License
This software is 100% written and maintained by myself (Adam Johnson) in my
free time. I claim and maintain full copyright in this software. You may
download, compile, modify and use the software for personal education or
curiosity. You may not release compiled versions of this software, nor
may you release modified versions of this code. You MAY NOT use any of this
code (neither text nor binary) for any commercial purposes or interests, nor
any commercial product.