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

uniform vec4 u_Cam;   // Camera Position passed to the shader
uniform vec4 u_Color; // The color with which to render this instance of geometry.
uniform vec4 u_Color2;
uniform float u_Time;

// These are the interpolated values out of the rasterizer, so you can't know
// their specific values without knowing the vertices that contributed to them
in vec4 fs_Pos;
in vec4 fs_Nor;
in vec4 fs_LightVec;
in vec4 fs_Col;

out vec4 out_Col; // This is the final output color that you will see on your
                  // screen for the pixel that is currently being processed.

////////////////////-------------- TOOLBOX FUNCTIONS --------------////////////////////
float getBias(float x, float bias) {
    return (x / ((((1.0f / bias) - 2.0f) * (1.0f - x)) + 1.0f));
}

float getGain(float x, float gain) {
    if(x < 0.5f)
        return getBias(x * 2.0f, gain) / 2.0f;
    else
        return getBias(x * 2.0f - 1.0f, 1.0f - gain) / 2.0f + 0.5f;
}

float easeInOutQuad(float x) {
    return x < 0.5f ? 2.0f * x * x : 1.0f - pow(-2.0f * x + 2.0f, 2.0f) / 2.0f;
}

float easeInQuad(float x) {
    return x * x;
}

float easeOutQuad(float x) {
    return 1.0f - (1.0f - x) * (1.0f - x);
}

float easeInOutCubic(float x) {
    return x < 0.5f ? 4.0f * x * x * x : 1.0f - pow(-2.0f * x + 2.0f, 3.0f) / 2.0f;
}
////////////////////-------------- NOISE FUNCTIONS --------------////////////////////

float noise3D(vec3 p) {
    return fract(sin(dot(p, vec3(127.1f, 311.7f, 213.f))) * 43758.5453f);
}

vec4 permute(vec4 x) {
    return mod(((x * 34.0f) + 1.0f) * x, 289.0f);
}
vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159f - 0.85373472095314f * r;
}
vec3 fade(vec3 t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float cnoise(vec3 P) {
    vec3 Pi0 = floor(P); // Integer part for indexing
    vec3 Pi1 = Pi0 + vec3(1.0f); // Integer part + 1
    Pi0 = mod(Pi0, 289.0f);
    Pi1 = mod(Pi1, 289.0f);
    vec3 Pf0 = fract(P); // Fractional part for interpolation
    vec3 Pf1 = Pf0 - vec3(1.0f); // Fractional part - 1.0
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;

    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);

    vec4 gx0 = ixy0 / 7.0f;
    vec4 gy0 = fract(floor(gx0) / 7.0f) - 0.5f;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5f) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0f));
    gx0 -= sz0 * (step(0.0f, gx0) - 0.5f);
    gy0 -= sz0 * (step(0.0f, gy0) - 0.5f);

    vec4 gx1 = ixy1 / 7.0f;
    vec4 gy1 = fract(floor(gx1) / 7.0f) - 0.5f;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5f) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0f));
    gx1 -= sz1 * (step(0.0f, gx1) - 0.5f);
    gy1 -= sz1 * (step(0.0f, gy1) - 0.5f);

    vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
    vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
    vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
    vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
    vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
    vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
    vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
    vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x;
    g011 *= norm1.y;
    g101 *= norm1.z;
    g111 *= norm1.w;

    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);

    vec3 fade_xyz = fade(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2f * n_xyz;
}

float interpNoise3D(float x, float y, float z) {
    int intX = int(floor(x));
    float fractX = fract(x);
    int intY = int(floor(y));
    float fractY = fract(y);
    int intZ = int(floor(z));
    float fractZ = fract(z);

    float v1 = noise3D(vec3(intX, intY, intZ));
    float v2 = noise3D(vec3(intX + 1, intY, intZ));
    float v3 = noise3D(vec3(intX, intY + 1, intZ));
    float v4 = noise3D(vec3(intX + 1, intY + 1, intZ));
    float v5 = noise3D(vec3(intX, intY, intZ + 1));
    float v6 = noise3D(vec3(intX + 1, intY, intZ + 1));
    float v7 = noise3D(vec3(intX, intY + 1, intZ + 1));
    float v8 = noise3D(vec3(intX + 1, intY + 1, intZ + 1));

    float i1 = mix(v1, v2, easeInOutQuad(fractX));
    float i2 = mix(v3, v4, easeInOutQuad(fractX));
    float i3 = mix(v5, v6, easeInOutQuad(fractX));
    float i4 = mix(v7, v8, easeInOutQuad(fractX));

    float m1 = mix(i1, i2, easeInOutQuad(fractY));
    float m2 = mix(i3, i4, easeInOutQuad(fractY));

    return mix(m1, m2, easeInOutQuad(fractZ));
}

float fbm(vec3 x) {
    float v = 0.0f;
    float a = 0.5f;
    vec3 shift = vec3(100);
    for(int i = 0; i < 5; ++i) {
        v += a * interpNoise3D(x.x, x.y, x.z);
        x = x * 2.0f + shift;
        a *= 0.5f;
    }
    return v;
}

void main() {

    // Material base color (before shading)
    vec4 diffuseColor = u_Color;
    vec4 topColor = u_Color2;

        // Calculate the diffuse term for Lambert shading
    float diffuseTerm = dot(normalize(fs_Nor), normalize(fs_LightVec));
        // Avoid negative lighting values
        // diffuseTerm = clamp(diffuseTerm, 0, 1);

    float ambientTerm = 0.99f;

    float lightIntensity = diffuseTerm + ambientTerm;   //Add a small float value to the color multiplier
                                                            //to simulate ambient lighting. This ensures that faces that are not
                                                            //lit by our point light are not completely black.

    // Dot product of camera direction and displaced vertex normal
    float direction = dot(normalize(fs_Nor.xyz), normalize(fs_Pos.xyz - u_Cam.xyz));

    //vec3((fs_Pos.y + 2.f) / 3.f)

        // Color based on height
    diffuseColor = vec4(mix(diffuseColor.xyz, topColor.xyz + (fbm(fs_Pos.xyz) * 0.33f), getBias(((fs_Pos.y) / (1.5f)), 0.27f + (abs(sin(u_Time / 10.f))) * 0.1f)), 1.f);

        // Calculating the outline
    if(direction <= 0.4f) {
        diffuseColor = vec4(0.f, 0.f, 0.f, 1.f);
    }

        // Compute final shaded color
    out_Col = vec4(diffuseColor.rgb, diffuseColor.a);
}
