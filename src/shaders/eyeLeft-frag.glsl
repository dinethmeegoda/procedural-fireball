#version 300 es

// This is a fragment shader. If you've opened this file first, please
// open and read lambert.vert.glsl before reading on.
// Unlike the vertex shader, the fragment shader actually does compute
// the shading of geometry. For every pixel in your program's output
// screen, the fragment shader is run for every bit of geometry that
// particular pixel overlaps. By implicitly interpolating the position
// data passed into the fragment shader by the vertex shader, the fragment shader
// can compute what color to apply to its pixel based on things like vertex
// position, light position, and vertex color.
precision highp float;

uniform vec4 u_Color; // The color with which to render this instance of geometry.
uniform vec4 u_Cam;

// These are the interpolated values out of the rasterizer, so you can't know
// their specific values without knowing the vertices that contributed to them
in vec4 fs_Pos;
in vec4 fs_Nor;
in vec4 fs_LightVec;
in vec4 fs_Col;
in vec4 fs_oNor;

out vec4 out_Col; // This is the final output color that you will see on your
                  // screen for the pixel that is currently being processed.

void main() {
    // Material base color (before shading)
  vec4 diffuseColor = u_Color;
  ;

        // Calculate the diffuse term for Lambert shading
  float diffuseTerm = dot(normalize(fs_Nor), normalize(fs_LightVec));
        // Avoid negative lighting values
        // diffuseTerm = clamp(diffuseTerm, 0, 1);

  float direction = dot(normalize(fs_Nor.xyz), normalize(fs_Pos.xyz - u_Cam.xyz));
  float direction2 = dot(normalize(fs_Pos.xyz - fs_oNor.xyz), normalize(u_Cam.xyz - fs_Pos.xyz));

  if(direction <= 0.6f) {
    diffuseColor = vec4(vec3(0.f, 0.f, 0.f), 1.f);
  } else if(direction2 >= 0.96f) {
    diffuseColor = diffuseColor = vec4(vec3(0.f, 0.f, 0.f), 1.f);
  }

        // Compute final shaded color
  out_Col = vec4(diffuseColor.rgb, diffuseColor.a);
}