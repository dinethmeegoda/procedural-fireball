#version 300 es
#define NUM_OCTAVES 5

//This is a vertex shader. While it is called a "shader" due to outdated conventions, this file
//is used to apply matrix transformations to the arrays of vertex data passed to it.
//Since this code is run on your GPU, each vertex is transformed simultaneously.
//If it were run on your CPU, each vertex would have to be processed in a FOR loop, one at a time.
//This simultaneous transformation allows your program to run much faster, especially when rendering
//geometry with millions of vertices.

uniform mat4 u_Model;       // The matrix that defines the transformation of the
                            // object we're rendering. In this assignment,
                            // this will be the result of traversing your scene graph.

uniform mat4 u_ModelInvTr;  // The inverse transpose of the model matrix.
                            // This allows us to transform the object's normals properly
                            // if the object has been non-uniformly scaled.

uniform mat4 u_ViewProj;    // The matrix that defines the camera's transformation.
                            // We've written a static matrix for you to use for HW2,
                            // but in HW3 you'll have to generate one yourself

uniform float u_Time;

in vec4 vs_Pos;             // The array of vertex positions passed to the shader

in vec4 vs_Nor;             // The array of vertex normals passed to the shader

in vec4 vs_Col;             // The array of vertex colors passed to the shader.

out vec4 fs_Nor;            // The array of normals that has been transformed by u_ModelInvTr. This is implicitly passed to the fragment shader.
out vec4 fs_LightVec;       // The direction in which our virtual light lies, relative to each vertex. This is implicitly passed to the fragment shader.
out vec4 fs_Col;            // The color of each vertex. This is implicitly passed to the fragment shader.
out vec4 fs_Pos;

const vec4 lightPos = vec4(5, 5, 3, 1); //The position of our virtual light, which is used to compute the shading of
                                        //the geometry in the fragment shader.

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
    fs_Col = vs_Col;                         // Pass the vertex colors to the fragment shader for interpolation

    fs_Pos = vs_Pos;

    mat3 invTranspose = mat3(u_ModelInvTr);
    fs_Nor = vec4(invTranspose * vec3(vs_Nor), 0);          // Pass the vertex normals to the fragment shader for interpolation.
                                                            // Transform the geometry's normals by the inverse transpose of the
                                                            // model matrix. This is necessary to ensure the normals remain
                                                            // perpendicular to the surface after the surface is transformed by
                                                            // the model matrix.

    fs_Pos.xyz *= -100.f;

    vec4 modelposition = u_Model * fs_Pos;   // Temporarily store the transformed vertex positions for use below

    fs_LightVec = lightPos - modelposition;  // Compute the direction in which the light source lies

    gl_Position = u_ViewProj * modelposition;// gl_Position is a built-in variable of OpenGL which is
                                             // used to render the final positions of the geometry's vertices
}
