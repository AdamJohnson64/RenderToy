#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform CONSTANTS
{
    mat4 vertexTransform;
};

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;

void main()
{
    gl_Position = vertexTransform * vec4(inPosition, 0, 1);
    outColor = inColor;
}