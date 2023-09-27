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

uniform float u_Time;       // Time passed to the shader

uniform vec4 u_Cam;         // Camera Direction passed to the shader

in vec4 vs_Pos;             // The array of vertex positions passed to the shader

in vec4 vs_Nor;             // The array of vertex normals passed to the shader

in vec4 vs_Col;             // The array of vertex colors passed to the shader.

out vec4 fs_Nor;            // The array of normals that has been transformed by u_ModelInvTr. This is implicitly passed to the fragment shader.
out vec4 fs_LightVec;       // The direction in which our virtual light lies, relative to each vertex. This is implicitly passed to the fragment shader.
out vec4 fs_Col;            // The color of each vertex. This is implicitly passed to the fragment shader.
out vec4 fs_Pos;

const vec4 lightPos = vec4(5, 5, 3, 1); //The position of our virtual light, which is used to compute the shading of
                                        //the geometry in the fragment shader.

////////////////////-------------- TOOLBOX FUNCTIONS --------------////////////////////
float getBias(float x, float bias) {
    return (x / ((((1.0f / bias) - 2.0f) * (1.0f - x)) + 1.0f));
}

float easeInOutQuad(float x) {
    return x < 0.5f ? 2.0f * x * x : 1.0f - pow(-2.0f * x + 2.0f, 2.0f) / 2.0f;
}

float easeOutQuad(float x) {
    return 1.0f - (1.0f - x) * (1.0f - x);
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

vec3 CalculateFlames(vec3 pos) {

    // Variables for transformation

    // The bigger lower level shapes
    float noise = cnoise(pos.xyz * 2.f + (u_Time / 60.f));

    float height = getBias((pos.y + 1.f) / 2.f, 0.1f);

    // Create shape on bottom half
    if(pos.y < 0.f) {
        float desiredY = pos.y * easeOutQuad(abs(pos.y + noise / 3.f)); //1.25f;
        pos.y = mix(pos.y, desiredY, height);
    }

    // Find verticle angle of the position
    float angleFromXZ = dot(pos - vec3(0.f, 0.f, 0.f), vec3(0.f, 1.f, 0.f));

    // Use Perlin Noise to compute flames
    float flames = abs(mix(pos.y, pos.y + noise, angleFromXZ)) * 1.2f + 0.4f;

    // Interpolate flames based on height
    pos.y = mix(pos.y, flames, height);

    return pos;

}

void main() {
    fs_Col = vs_Col;                         // Pass the vertex colors to the fragment shader for interpolation

    mat3 invTranspose = mat3(u_ModelInvTr);
    fs_Nor = vec4(invTranspose * vec3(vs_Nor), 0);          // Pass the vertex normals to the fragment shader for interpolation.
                                                            // Transform the geometry's normals by the inverse transpose of the
                                                            // model matrix. This is necessary to ensure the normals remain
                                                            // perpendicular to the surface after the surface is transformed by
                                                            // the model matrix.

    fs_Pos = vs_Pos;

    // Trying to recalculate normals after transformation
    vec3 randomVec;
    // If the random vector is not parallel or anti-parallel to the normal vector
    if(1.f - abs(dot(vs_Nor.xyz, vec3(0.f, 1.f, 1.f))) < .1f) {
        randomVec = vec3(0.f, 1.f, 1.f);
    } else {
        randomVec = vec3(1.f, 1.f, 0.f);
    }

    // Use cross product to figure out BiTangent and Tangent vectors to the normal
    vec3 biTangent = normalize(cross(vs_Nor.xyz, randomVec));
    vec3 tangent = normalize(cross(vs_Nor.xyz, biTangent));

    // find new points along the tangent and bitangent vectors
    vec3 p1 = fs_Pos.xyz + (biTangent * 0.001f);
    vec3 p2 = fs_Pos.xyz + (tangent * 0.001f);

    // transform all 3 points
    fs_Pos = vec4(CalculateFlames(fs_Pos.xyz), 1.f);
    p1 = CalculateFlames(p1);
    p2 = CalculateFlames(p2);

    // calculate the new normal
    fs_Nor = vec4(normalize(cross(fs_Pos.xyz - p1, p2 - fs_Pos.xyz)), 1.f);

    vec4 modelposition = u_Model * fs_Pos;   // Temporarily store the transformed vertex positions for use below

    fs_LightVec = lightPos - modelposition;  // Compute the direction in which the light source lies

    gl_Position = u_ViewProj * modelposition;// gl_Position is a built-in variable of OpenGL which is
                                             // used to render the final positions of the geometry's vertices
}
