#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 inColor;

layout (set = 1, binding = 0) uniform texture2D TEXTURE;
layout (set = 2, binding = 0) uniform sampler SAMPLER;

layout(location = 0) out vec4 outColor;

void main()
{
    float x = gl_FragCoord.x / 256.0f;
    float y = gl_FragCoord.y / 256.0f;
    outColor = inColor;
    outColor.rg = texture(sampler2D(TEXTURE, SAMPLER), vec2(x, y)).rg;
}