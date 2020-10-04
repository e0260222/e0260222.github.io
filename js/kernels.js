const canvasParent = document.getElementById('canvas-parent');

const horizontalFlip = document.getElementById('horizontal-flip');
const verticalFlip = document.getElementById('vertical-flip');
const sharpen = document.getElementById('sharpen');
const gaussianBlur = document.getElementById('gaussian-blur');
const invertColours = document.getElementById('invert-colours');
const grayscaleColours = document.getElementById('grayscale-colours');
const edgeDetection = document.getElementById('edge-detection');

const enableGpu = document.getElementById('enable-gpu');
const fpsNumber = document.getElementById('fps-number');
const timeTaken = document.getElementById('time-taken');
let lastCalledTime = Date.now();
let fps;
let delta;
let dispose = setup();
enableGpu.onchange = () => {
    if (dispose)
        dispose();
    dispose = setup();
};
function setup() {
    let disposed = false;
    const gpu = new GPU({
        mode: enableGpu.checked ? 'gpu' : 'cpu'
    });

    // THIS IS THE IMPORTANT STUFF
    const horizontalFlipKernal = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        if (isEnabled) {
            const pixel2 = frame[this.thread.y][640 - 1 - this.thread.x];   
            this.color(pixel2.r, pixel2.g, pixel2.b, pixel2.a);
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
    
     const verticalFlipKernal = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        if (isEnabled) {
            const pixel2 = frame[480 - 1 - this.thread.y][this.thread.x];   
            this.color(pixel2.r, pixel2.g, pixel2.b, pixel2.a);
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
    
    const invertColourKernel = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        if (isEnabled) {
            this.color(1 - pixel.r, 1 - pixel.g, 1 - pixel.b, pixel.a);
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
    
    const grayscaleColourKernal = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        if (isEnabled) {
            const grey = Math.fround(0.3 * pixel.r + 0.59 * pixel.g + 0.11 * pixel.b);
            this.color(grey, grey, grey, pixel.a);
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
    
    const sharpenKernel = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        var col = [0, 0, 0];
        if (isEnabled) {
            const k0 = 0;
            const k1 = -1;
            const k2 = 0;
            const k3 = -1;
            const k4 = 5;
            const k5 = -1;
            const k6 = 0;
            const k7 = -1;
            const k8 = 0;
            if (this.thread.y > 0 && this.thread.y < 480 - 1 && this.thread.x < 640 - 1 && this.thread.x >0) {
                const a0 = frame[this.thread.y + 1][this.thread.x - 1];
                const a1 = frame[this.thread.y + 1][this.thread.x];
                const a2 = frame[this.thread.y + 1][this.thread.x + 1];
                const a3 = frame[this.thread.y][this.thread.x - 1];
                const a4 = frame[this.thread.y][this.thread.x];
                const a5 = frame[this.thread.y][this.thread.x + 1];
                const a6 = frame[this.thread.y - 1][this.thread.x - 1];
                const a7 = frame[this.thread.y - 1][this.thread.x];
                const a8 = frame[this.thread.y - 1][this.thread.x + 1];
                for (var i = 0; i < 3; i++) { // Compute the convolution for each of red [0], green [1] and blue [2]
                    col[i] = a0[i] * k0 + a1[i] * k1 + a2[i] * k2 + a3[i] * k3 + a4[i]* k4 
                                + a5[i] * k5 + a6[i] * k6 + a7[i] * k7 + a8[i] * k8;
                }
                this.color(col[0], col[1], col[2], 1);
            } else {
                this.color(pixel.r, pixel.g, pixel.b, pixel.a);
            }
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
    
    const gaussianBlurKernel = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        var col = [0, 0, 0];
        if (isEnabled) {
            const k0 = 1 / 16;
            const k1 = 2 / 16;
            const k2 = 1 / 16;
            const k3 = 2 / 16;
            const k4 = 4 / 16;
            const k5 = 2 / 16;
            const k6 = 1 / 16;
            const k7 = 2 / 16;
            const k8 = 1 / 16;
            if (this.thread.y > 0 && this.thread.y < 480 - 1 && this.thread.x < 640 - 1 && this.thread.x >0) {
                const a0 = frame[this.thread.y + 1][this.thread.x - 1];
                const a1 = frame[this.thread.y + 1][this.thread.x];
                const a2 = frame[this.thread.y + 1][this.thread.x + 1];
                const a3 = frame[this.thread.y][this.thread.x - 1];
                const a4 = frame[this.thread.y][this.thread.x];
                const a5 = frame[this.thread.y][this.thread.x + 1];
                const a6 = frame[this.thread.y - 1][this.thread.x - 1];
                const a7 = frame[this.thread.y - 1][this.thread.x];
                const a8 = frame[this.thread.y - 1][this.thread.x + 1];
                for (var i = 0; i < 3; i++) { // Compute the convolution for each of red [0], green [1] and blue [2]
                    col[i] = a0[i] * k0 + a1[i] * k1 + a2[i] * k2 + a3[i] * k3 + a4[i]* k4 
                                + a5[i] * k5 + a6[i] * k6 + a7[i] * k7 + a8[i] * k8;
                }
                this.color(col[0], col[1], col[2], 1);
            } else {
                this.color(pixel.r, pixel.g, pixel.b, pixel.a);
            }
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
    
    const edgeDetectionKernel = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        var col = [0, 0, 0];
        if (isEnabled) {
            const k0 = -1;
            const k1 = -1;
            const k2 = -1;
            const k3 = -1;
            const k4 = 8;
            const k5 = -1;
            const k6 = -1;
            const k7 = -1;
            const k8 = -1;
            if (this.thread.y > 0 && this.thread.y < 480 - 1 && this.thread.x < 640 - 1 && this.thread.x >0) {
                const a0 = frame[this.thread.y + 1][this.thread.x - 1];
                const a1 = frame[this.thread.y + 1][this.thread.x];
                const a2 = frame[this.thread.y + 1][this.thread.x + 1];
                const a3 = frame[this.thread.y][this.thread.x - 1];
                const a4 = frame[this.thread.y][this.thread.x];
                const a5 = frame[this.thread.y][this.thread.x + 1];
                const a6 = frame[this.thread.y - 1][this.thread.x - 1];
                const a7 = frame[this.thread.y - 1][this.thread.x];
                const a8 = frame[this.thread.y - 1][this.thread.x + 1];
                for (var i = 0; i < 3; i++) { // Compute the convolution for each of red [0], green [1] and blue [2]
                    col[i] = a0[i] * k0 + a1[i] * k1 + a2[i] * k2 + a3[i] * k3 + a4[i]* k4 
                                + a5[i] * k5 + a6[i] * k6 + a7[i] * k7 + a8[i] * k8;
                }
                this.color(col[0], col[1], col[2], 1);
            } else {
                this.color(pixel.r, pixel.g, pixel.b, pixel.a);
            }
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
            
    const textureToImageKernel = gpu.createKernel(function (frame) {
        const pixel = frame[this.thread.y][this.thread.x];
        this.color(pixel.r, pixel.g, pixel.b, pixel.a);
    }, {
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
        
    canvasParent.appendChild(textureToImageKernel.canvas);
    const videoElement = document.querySelector('video');  
    function render() {
        if (disposed) {
            return;
        }
        
        if (enableGpu.checked){
            var t0 = performance.now();
            const outputStage1 = horizontalFlipKernal(videoElement, horizontalFlip.checked);
            const outputStage2 = verticalFlipKernal(outputStage1, verticalFlip.checked);
            const outputStage3 = sharpenKernel(outputStage2, sharpen.checked);
            const outputStage4 = gaussianBlurKernel(outputStage3, gaussianBlur.checked);
            const outputStage5 = invertColourKernel(outputStage4, invertColours.checked);
            const outputStage6 = grayscaleColourKernal(outputStage5, grayscaleColours.checked);
            const outputStage7 = edgeDetectionKernel(outputStage6, edgeDetection.checked);
            var t1 = performance.now();
            var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(0);
                    console.log(timeTaken.innerHTML);
            textureToImageKernel(outputStage7);
        } else {
            switch (getRadioValue()) {
                case "horizontal-flip":
                    var t0 = performance.now();
                    horizontalFlipKernal(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(0);
                    break;
                case "vertical-flip":
                    verticalFlipKernal(videoElement, true);
                    break;
                case "sharpen":
                    sharpenKernel(videoElement, true);
                    break;
                case "gaussian-blur":
                    var t0 = performance.now();
                    gaussianBlurKernel(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(0);
                    break;
                case "invert-colours":
                    invertColourKernel(videoElement, true);
                    break;
                case "grayscale-colours":
                    grayscaleColourKernal(videoElement, true);
                    break;
                case "edge-detection":
                    edgeDetectionKernel(videoElement, true);
                    break;
                case "no-filter":
                    var t0 = performance.now();
                    textureToImageKernel(videoElement);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(0);
            }                        
        }
                
        window.requestAnimationFrame(render);
        calcFPS();
    }

    render();
    return () => {
        canvasParent.removeChild(textureToImageKernel.canvas);
        gpu.destroy();
        disposed = true;
    };
}

function streamHandler(stream) {
    try {
        video.srcObject = stream;
    } catch (error) {
        video.src = URL.createObjectURL(stream);
    }
    video.play();
    console.log("In startStream");
    requestAnimationFrame(render);
}

addEventListener("DOMContentLoaded", initialize);

function calcFPS() {
    delta = (Date.now() - lastCalledTime) / 1000;
    lastCalledTime = Date.now();
    fps = 1 / delta;
    fpsNumber.innerHTML = fps.toFixed(0);
}

function calcTimeTaken(delta) {
    
    timeTaken.innerHTML = delta;
}

function toggleDiv(){
    if (enableGpu.checked == true){
        document.getElementById("gpu-controls").style.display = "block";
        document.getElementById("cpu-controls").style.display = "none";
    } else {
        document.getElementById("gpu-controls").style.display = "none";
        document.getElementById("cpu-controls").style.display = "block";
    }
}

function getRadioValue() {
    var elements = document.getElementsByName('filter');    
    for(i = 0; i < elements.length; i++) { 
        if(elements[i].checked) {
            return elements[i].value;
        }
    } 
}