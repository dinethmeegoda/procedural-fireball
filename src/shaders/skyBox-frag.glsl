#version 300 es
#define NUM_OCTAVES 5

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
uniform float u_Time;

// These are the interpolated values out of the rasterizer, so you can't know
// their specific values without knowing the vertices that contributed to them
in vec4 fs_Nor;
in vec4 fs_LightVec;
in vec4 fs_Col;
in vec4 fs_Pos;

out vec4 out_Col; // This is the final output color that you will see on your
                  // screen for the pixel that is currently being processed.

float mod289(float x) {
    return x - floor(x * (1.0f / 289.0f)) * 289.0f;
}
vec4 mod289(vec4 x) {
    return x - floor(x * (1.0f / 289.0f)) * 289.0f;
}
vec4 perm(vec4 x) {
    return mod289(((x * 34.0f) + 1.0f) * x);
}

float noise(vec3 p) {
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0f - 2.0f * d);

    vec4 b = a.xxyy + vec4(0.0f, 1.0f, 0.0f, 1.0f);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0f);

    vec4 o1 = fract(k3 * (1.0f / 41.0f));
    vec4 o2 = fract(k4 * (1.0f / 41.0f));

    vec4 o3 = o2 * d.z + o1 * (1.0f - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0f - d.x);

    return o4.y * d.y + o4.x * (1.0f - d.y);
}

float fbm(vec3 x) {
    float v = 0.0f;
    float a = 0.5f;
    vec3 shift = vec3(100);
    for(int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(x);
        x = x * 2.0f + shift;
        a *= 0.5f;
    }
    return v;
}

vec3 hash33(vec3 p3) {
    vec3 p = fract(p3 * vec3(.1031f, .11369f, .13787f));
    p += dot(p, p.yxz + 19.19f);
    return -1.0f + 2.0f * fract(vec3((p.x + p.y) * p.z, (p.x + p.z) * p.y, (p.y + p.z) * p.x));
}

float worley(vec3 p, float scale) {

    vec3 id = floor(p * scale);
    vec3 fd = fract(p * scale);

    float n = 0.f;

    float minimalDist = 1.f;

    for(float x = -1.f; x <= 1.f; x++) {
        for(float y = -1.f; y <= 1.f; y++) {
            for(float z = -1.f; z <= 1.f; z++) {

                vec3 coord = vec3(x, y, z);
                vec3 rId = hash33(mod(id + coord, scale)) * 0.5f + 0.5f;

                vec3 r = coord + rId - fd;

                float d = dot(r, r);

                if(d < minimalDist) {
                    minimalDist = d;
                }

            }//z
        }//y
    }//x

    return 1.0f - minimalDist;
}

float getBias(float x, float bias) {
    return (x / ((((1.0f / bias) - 2.0f) * (1.0f - x)) + 1.0f));
}

void main() {
    // Material base color (before shading)
    vec4 diffuseColor = u_Color;

    vec4 pos = vec4(normalize(fs_Pos.xyz), 1.f);

    float noise = worley(pos.xyz + 2.f * getBias(cos(u_Time / 100.f), 0.2f), worley(pos.xyz + u_Time / 1000.f, 2.f));

    diffuseColor = vec4(noise, 0.5f, 0.7f, 1) - vec4(0.94f, noise, noise, 0) + u_Color;
        // diffuseColor =  vec4(u_Color.rgb * (1. - noise), 1.);

        // diffuseColor = mix(u_Color, vec4(1), noise);

        // Calculate the diffuse term for Lambert shading
    float diffuseTerm = dot(normalize(fs_Nor), normalize(fs_LightVec));
        // Avoid negative lighting values
        // diffuseTerm = clamp(diffuseTerm, 0, 1);

    float ambientTerm = 0.99f;

    float lightIntensity = diffuseTerm + ambientTerm;   //Add a small float value to the color multiplier
                                                            //to simulate ambient lighting. This ensures that faces that are not
                                                            //lit by our point light are not completely black.

        // Compute final shaded color
    out_Col = vec4(diffuseColor.rgb * lightIntensity, diffuseColor.a);
}
