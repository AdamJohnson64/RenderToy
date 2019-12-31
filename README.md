# RenderToy

## Inspiration
3D Graphics is a huge topic. Back when I started studying this in the 90s some
of the concepts were as immature as my view of them. My first rasterizer was
in screen space and interpolated incorrectly (without the homogeneous divide)
and I didn't even know what linear algebra was.

This project is a refresh as I build a sample project mostly as a teaching aid
for people that want to learn about 3D and how it works under the hood. A key
feature is that everything I touch on will have a software reference
implementation so you can see the math behind the hardware renderer.

I also want to look at non-realtime rendering by taking a walk through
Raytrace Lane.

Primary software goals are:
- Simplicity.
  - Everything simple, brief and self-explanatory.
  - Performance is always good but it's not our major concern here.
- Flexibility.
  - I'm not building an OpenGL state machine; we want to drop in concepts
    such that we can tune out things we don't want to think about.
  - Deal with each concept in isolation so we can demonstrate it.
- Reusability.
  - Parts of this can be a tooling platform for trying other things.
  - Split out controls (etc) for the express purpose of simple reuse.

Personally I'm a huge fan of build infrastructure and software designs that
serve that purpose. Code should be easy to build and easy to test so we take
a little extra time here to get that structure right.

## Latest
One thing we all end up doing at some point is using a graphics abstraction to
handle things like D3D12 vs Vulkan so the goal here is to build such an
abstraction. This is one of the goals for "Arcturus", along with more C++ code
so we can handle more architectures and targets.

## License
Code presented here within this repository is 100% written by myself
(Adam Johnson) in my free time, and I claim full copyright in this software.

Code referenced via submodules should be referred to as 3rd Party Software
and all rights and licensing terms remain the property of their respective
owners.

This repository does NOT confer any sublicense to 3rd Party components. If you
wish to use submodules of this code you shall adhere to the independent license
terms of those submodule components.

You may download, compile, modify and use the software for personal education
or curiosity.

You MAY NOT release compiled versions of this software, nor may you release
modified versions of this code, neither in part nor whole.

You MAY NOT use any of this code (neither text nor binary) for any commercial
purposes or interests, nor any commercial product, neither in part nor whole.