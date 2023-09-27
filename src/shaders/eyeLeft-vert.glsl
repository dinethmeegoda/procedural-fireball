#version 300 es

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
out vec4 fs_oNor;

const vec4 lightPos = vec4(5, 5, 3, 1); //The position of our virtual light, which is used to compute the shading of
                                        //the geometry in the fragment shader.

vec3 calculateTransformation(vec3 pos) {

  mat3 rot = mat3(vec3(0.f, 0.f, -1.f), vec3(0.f, 1.f, 0.f), vec3(1.f, 0.f, 1.f));

  pos = rot * pos;

    // Transforming the eye to the right place
  pos = vec3(pos.xyz / 5.f);

  pos = vec3(pos.x, (pos.y / 1.2f), pos.z / 2.5f);

  pos.z += 1.05f;

  pos.x -= 0.3f;

  pos.y += abs(sin((u_Time + 100.f) / 50.f) * 0.05f);

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
  fs_Pos = vec4(calculateTransformation(fs_Pos.xyz), 1.f);
  p1 = calculateTransformation(p1);
  p2 = calculateTransformation(p2);

  //transform center of sphere
  vec3 center = vec3(0.25f, -0.25f, 0.f);
  center = calculateTransformation(center);
  fs_oNor = vec4(center, 1.f);

  // calculate the new normal
  fs_Nor = vec4(normalize(cross(fs_Pos.xyz - p1, p2 - fs_Pos.xyz)), 1.f);

  vec4 modelposition = u_Model * fs_Pos;   // Temporarily store the transformed vertex positions for use below

  fs_LightVec = lightPos - modelposition;  // Compute the direction in which the light source lies

  gl_Position = u_ViewProj * modelposition;// gl_Position is a built-in variable of OpenGL which is
                                             // used to render the final positions of the geometry's vertices
}