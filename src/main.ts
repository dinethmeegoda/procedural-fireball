import {vec3, vec4} from 'gl-matrix';
const Stats = require('stats-js');
import * as DAT from 'dat.gui';
import Icosphere from './geometry/Icosphere';
import Square from './geometry/Square';
import Cube from './geometry/Cube';
import OpenGLRenderer from './rendering/gl/OpenGLRenderer';
import Camera from './Camera';
import {setGL} from './globals';
import ShaderProgram, {Shader} from './rendering/gl/ShaderProgram';

// Define an object with application parameters and button callbacks
// This will be referred to by dat.GUI's functions that add GUI elements.
const controls = {
  tesselations: 5,
  color: [175, 46, 29] as [number, number, number],
  secondary_color: [255, 161, 101.5] as [number, number, number],
  eye_color: [255, 255, 0] as [number, number, number],
};

// true = cube, false = icosphere, it gets swapped at initialize 
let shape = true;
let icosphere: Icosphere;
let square: Square;
let cube: Cube;
let prevTesselations: number = 5;
let prevColor = [0, 0, 0] as [number, number, number];
let prevSecondColor = [0, 0, 0] as [number, number, number];
let prevEyeColor = [0, 0, 0] as [number, number, number];
let time = 0;

function loadScene() {
  icosphere = new Icosphere(vec3.fromValues(0, 0, 0), 1, controls.tesselations);
  icosphere.create();
  cube = new Cube(1.5);
  cube.create();
  shape = !shape;
}

function main() {
  // Initial display for framerate
  const stats = Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0px';
  stats.domElement.style.top = '0px';
  document.body.appendChild(stats.domElement);

  // Add controls to the gui
  const gui = new DAT.GUI();
  gui.add(controls, 'tesselations', 0, 8).step(1);
  gui.addColor(controls, 'color');
  gui.addColor(controls, 'secondary_color');
  gui.addColor(controls, 'eye_color');

  // get canvas and webgl context
  const canvas = <HTMLCanvasElement> document.getElementById('canvas');
  const gl = <WebGL2RenderingContext> canvas.getContext('webgl2');
  if (!gl) {
    alert('WebGL 2 not supported!');
  }
  // `setGL` is a function imported above which sets the value of `gl` in the `globals.ts` module.
  // Later, we can import `gl` from `globals.ts` to access it
  setGL(gl);

  // Initial call to load scene
  loadScene();

  const camera = new Camera(vec3.fromValues(0, 0, 5), vec3.fromValues(0, 0, 0));

  const renderer = new OpenGLRenderer(canvas);
  renderer.setClearColor(0.2, 0.2, 0.2, 1);
  gl.enable(gl.DEPTH_TEST);

  const fire = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/fire-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/fire-frag.glsl')),
  ]);

  const eyeLeft = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/eyeLeft-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/eyeLeft-frag.glsl')),
  ]);

  const eyeRight = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/eyeRight-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/eyeRight-frag.glsl')),
  ]);

  const sky = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/skyBox-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/skyBox-frag.glsl')),
  ]);

  // This function will be called every frame
  function tick() {
    camera.update();
    stats.begin();
    gl.viewport(0, 0, window.innerWidth, window.innerHeight);
    renderer.clear();
    fire.setTime(time);
    eyeRight.setTime(time);
    eyeLeft.setTime(time);
    sky.setTime(time);
    if(controls.tesselations != prevTesselations)
    {
      prevTesselations = controls.tesselations;
      icosphere = new Icosphere(vec3.fromValues(0, 0, 0), 1, prevTesselations);
      icosphere.create();
    }
    if (!vec3.equals(controls.color, prevColor)) {
      prevColor = controls.color;
      const newColor = vec4.fromValues(...prevColor, 256);
      vec4.scale(newColor, newColor, 1 / 256);
      fire.setGeometryColor(newColor);
    }
    if (!vec3.equals(controls.secondary_color, prevSecondColor)) {
      prevSecondColor = controls.secondary_color;
      const newSecondColor = vec4.fromValues(...prevSecondColor, 256);
      vec4.scale(newSecondColor, newSecondColor, 1 / 256);
      fire.setSecondaryColor(newSecondColor);
    }
    if (!vec3.equals(controls.eye_color, prevEyeColor)) {
      prevEyeColor = controls.eye_color;
      const newEyeColor = vec4.fromValues(...prevEyeColor, 256);
      vec4.scale(newEyeColor, newEyeColor, 1 / 256);
      eyeLeft.setGeometryColor(newEyeColor);
      eyeRight.setGeometryColor(newEyeColor);
    }
    fire.setCam(vec4.fromValues(camera.controls.eye[0], camera.controls.eye[1], camera.controls.eye[2], 1));
    eyeLeft.setCam(vec4.fromValues(camera.controls.eye[0], camera.controls.eye[1], camera.controls.eye[2], 1));
    eyeRight.setCam(vec4.fromValues(camera.controls.eye[0], camera.controls.eye[1], camera.controls.eye[2], 1));
    const renderedObj = shape ? cube : icosphere;
    renderer.render(camera, sky, [
      cube,
    ]);
    renderer.render(camera, fire, [
      renderedObj,
    ]);
    renderer.render(camera, eyeLeft, [
      icosphere,
    ]);
    renderer.render(camera, eyeRight, [
      icosphere,
    ]);
    stats.end();

    // Tell the browser to call `tick` again whenever it renders a new frame
    time++;
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', function() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.setAspectRatio(window.innerWidth / window.innerHeight);
    camera.updateProjectionMatrix();
  }, false);

  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.setAspectRatio(window.innerWidth / window.innerHeight);
  camera.updateProjectionMatrix();

  // Start the render loop
  tick();
}

main();
