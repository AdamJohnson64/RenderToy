# RenderToy

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