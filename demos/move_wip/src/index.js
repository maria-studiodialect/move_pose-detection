/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';
import * as mpPose from '@mediapipe/pose';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';

import { Camera } from './camera';
import { setupDatGui } from './option_panel';
import { STATE } from './params';
import { setupStats } from './stats_panel';
import { setBackendAndEnvFlags } from './util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
let poses;


async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: 500, height: 500 },
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
          STATE.model, { runtime, modelType: STATE.modelConfig.type });
      }
    case posedetection.SupportedModels.MoveNet:
      let modelType;
      if (STATE.modelConfig.type == 'lightning') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
      } else if (STATE.modelConfig.type == 'thunder') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      } else if (STATE.modelConfig.type == 'multipose') {
        modelType = posedetection.movenet.modelType.MULTIPOSE_LIGHTNING;
      }
      const modelConfig = { modelType };

      if (STATE.modelConfig.customModel !== '') {
        modelConfig.modelUrl = STATE.modelConfig.customModel;
      }
      if (STATE.modelConfig.type === 'multipose') {
        modelConfig.enableTracking = STATE.modelConfig.enableTracking;
      }
      return posedetection.createDetector(STATE.model, modelConfig);
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
      1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  //let poses = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimatePoses.
    beginEstimatePosesStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      poses = await detector.estimatePoses(
        camera.video,
        { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimatePosesStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (poses && poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
    usePoses(poses);
  }

}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }
  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    //alert('Cannot find model in the query string.');
    urlParams.set('model', 'movenet');
    window.location.search = urlParams;
    return;
  }
  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  renderPrediction();
};


app();



const usePoses = (poses) => {
  const Body = m.Body;
  const attractiveBody = m.attractiveBody;
  const poseX = poses[0].keypoints[0].x;
  const poseY = poses[0].keypoints[0].y;
  Body.translate(attractiveBody, {
    x: (poseX - attractiveBody.position.x),
    y: (poseY - attractiveBody.position.y)
  });

}



var canvas = $("#wrapper-canvas").get(0);

var dimensions = {
  width: 1280,
  height: 720
};

Matter.use('matter-attractors');
Matter.use('matter-wrap');



function runMatter() {
  // module aliases
  var Engine = Matter.Engine,
    Events = Matter.Events,
    Runner = Matter.Runner,
    Render = Matter.Render,
    World = Matter.World,
    Body = Matter.Body,
    Mouse = Matter.Mouse,
    Common = Matter.Common,
    Composite = Matter.Composite,
    Composites = Matter.Composites,
    Bodies = Matter.Bodies;

  // create engine
  var engine = Engine.create();

  engine.world.gravity.y = 0
  engine.world.gravity.x = 0
  engine.world.gravity.scale = 0.1

  // create renderer
  var render = Render.create({
    element: canvas,
    engine: engine,
    options: {
      showVelocity: false,
      width: dimensions.width,
      height: dimensions.height,
      wireframes: false,
      background: '#B0B1B3'
    }
  });

  // create runner
  var runner = Runner.create();

  // Runner.run(runner, engine);
  // Render.run(render);

  // create demo scene
  var world = engine.world;
  world.gravity.scale = 0;

  // create a body with an attractor
  var attractiveBody = Bodies.circle(
    render.options.width / 2,
    render.options.height / 2,
    (Math.max(dimensions.width / 10, dimensions.height / 10)) / 2,
    {
      render: {
        fillStyle: `#B0B1B3`,
        strokeStyle: `gray`,
        lineWidth: 1
      },
      isStatic: true,
      plugin: {
        attractors: [
          function (bodyA, bodyB) {
            return {
              x: (bodyA.position.x - bodyB.position.x) * 1e-6,
              y: (bodyA.position.y - bodyB.position.y) * 1e-6,
            };
          }
        ]
      }
    });

  World.add(world, attractiveBody);

  // add some bodies that to be attracted
  for (var i = 0; i < 60; i += 1) {
    /*
    let x = Common.random(0, render.options.width);
    let y = Common.random(0, render.options.height);
    let s = Common.random() > 0.6 ? Common.random(10, 80) : Common.random(4, 60);
    let poligonNumber = Common.random(3, 6);
    var body = Bodies.polygon(
      x, y,
      poligonNumber,
      s,

      {
        mass: s / 20,
        friction: 0,
        frictionAir: 0.02,
        angle: Math.round(Math.random() * 360),
        render: {
          fillStyle: '#FFFFFF',
          strokeStyle: `#DDDDDD`,
          lineWidth: 2
        }
      }
    );

    World.add(world, body);


    let r = Common.random(0, 1)
    var circle = Bodies.circle(x, y, Common.random(2, 8), {
      mass: 0.1,
      friction: 0,
      frictionAir: 0.01,
      render: {
        fillStyle: r > 0.3 ? `#FF2D6A` : `rgb(240,240,240)`,
        strokeStyle: `#E9202E`,
        lineWidth: 2
      }
    });
    World.add(world, circle);

    var circle = Bodies.circle(x, y, Common.random(2, 20), {
      mass: 6,
      friction: 0,
      frictionAir: 0,
      render: {
        fillStyle: r > 0.3 ? `#4267F8` : `rgb(240,240,240)`,
        strokeStyle: `#3257E8`,
        lineWidth: 4
      }
    });
    World.add(world, circle);

    var circle = Bodies.circle(x, y, Common.random(2, 30), {
      mass: 0.2,
      friction: 0.6,
      frictionAir: 0.8,
      render: {
        fillStyle: `rgb(240,240,240)`,
        strokeStyle: `#FFFFFF`,
        lineWidth: 3
      }
    });
    World.add(world, circle);
    */
  }

  var radius = 20
  // art & design
  var illustration = Bodies.rectangle(70, 500, 237, 80, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/01.png', xScale: 0.5, yScale: 0.5 } } })
  var art = Bodies.rectangle(35, 460, 288, 75, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/02.png', xScale: 0.5, yScale: 0.5 } } })
  var threeD = Bodies.rectangle(90, 460, 307, 59, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/03.png', xScale: 0.5, yScale: 0.5 } } })
  var graphic = Bodies.rectangle(60, 420, 223, 60, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/04.png', xScale: 0.5, yScale: 0.5 } } })
  var photo = Bodies.rectangle(50, 380, 174, 62, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/05.png', xScale: 0.5, yScale: 0.5 } } })
  // video
  var documentary = Bodies.rectangle(220, 540, 238, 59, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/06.png', xScale: 0.5, yScale: 0.5 } } })
  var animation = Bodies.rectangle(200, 490, 200, 70, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/07.png', xScale: 0.5, yScale: 0.5 } } })
  var play = Bodies.rectangle(190, 440, 208, 71, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/08.png', xScale: 0.5, yScale: 0.5 } } })
  var climb = Bodies.rectangle(190, 440, 249, 62, { chamfer: { radius: radius }, mass: 0.1, friction: 0, frictionAir: 0.01, render: { sprite: { texture: 'https://maria-studiodialect.github.io/hosted-assets/09.png', xScale: 0.5, yScale: 0.5 } } })


  // add all of the bodies to the world
  World.add(world, [
    illustration, art, threeD, graphic, photo, documentary, animation, play, climb
  ]);
  // add mouse control
  var mouse = Mouse.create(render.canvas);

  Events.on(engine, 'afterUpdate', function () {
    if (!mouse.position.x) return;
    // smoothly move the attractor body towards the mouse

  });

  // return a context for MatterDemo to control
  let data = {
    attractiveBody,
    Body,
    engine: engine,
    runner: runner,
    render: render,
    canvas: render.canvas,
    stop: function () {
      Matter.Render.stop(render);
      Matter.Runner.stop(runner);
    },
    play: function () {
      Matter.Runner.run(runner, engine);
      Matter.Render.run(render);
    }
  };
  Matter.Runner.run(runner, engine);
  Matter.Render.run(render);
  return data;
}

function debounce(func, wait, immediate) {
  var timeout;
  return function () {
    var context = this, args = arguments;
    var later = function () {
      timeout = null;
      if (!immediate) func.apply(context, args);
    };
    var callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    if (callNow) func.apply(context, args);
  };
};

function setWindowSize() {
  let dimensions = {};
  dimensions.width = 1280;
  dimensions.height = 720;

  m.render.canvas.width = 1280;
  m.render.canvas.height = 720;
  return dimensions;
}

let m = runMatter()
setWindowSize()
$(window).resize(debounce(setWindowSize, 250))


