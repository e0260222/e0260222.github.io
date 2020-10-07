const canvasParent = document.getElementById('canvas-parent');

const horizontalFlip = document.getElementById('horizontal-flip');
const verticalFlip = document.getElementById('vertical-flip');
const sharpen = document.getElementById('sharpen');
const gaussianBlur = document.getElementById('gaussian-blur');
const invertColours = document.getElementById('invert-colours');
const grayscaleColours = document.getElementById('grayscale-colours');
const edgeDetection = document.getElementById('edge-detection');
const encrypt = document.getElementById('encrypt');
const decrypt = document.getElementById('decrypt');

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
    
    // Kernel for performing horizontal flip
    const horizontalFlipKernal = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        if (isEnabled) {
            //Performs pixel swapping
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
    
    // Kernel for performing vertical flip
    const verticalFlipKernal = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        if (isEnabled) {
            //Performs pixel swapping
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
    
    // Kernel for performing colours invert
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
    
    // Kernel for performing grayscaling
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
    
    // Kernel for performing image sharpening
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
            if (this.thread.y > 0 && this.thread.y < 480 - 1 && this.thread.x < 640 - 1 && this.thread.x > 0) {
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
                this.color(col[0], col[1], col[2], pixel.a);
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
    
    // Kernel for performing image blurring
    const gaussianBlurKernel = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        var col = [0, 0, 0];
        if (isEnabled) {
            const k0 = 1 / 256;
            const k1 = 4 / 256;
            const k2 = 6 / 256;
            const k3 = 4 / 256;
            const k4 = 1 / 256;            
            const k5 = 4 / 256;
            const k6 = 16 / 256;
            const k7 = 24 / 256;
            const k8 = 16 / 256;
            const k9 = 4 / 256;            
            const k10 = 6 / 256;
            const k11 = 24 / 256;
            const k12 = 36 / 256;
            const k13 = 24 / 256;
            const k14 = 6 / 256;            
            const k15 = 4 / 256;
            const k16 = 16 / 256;
            const k17 = 24 / 256;
            const k18 = 16 / 256;
            const k19 = 4 / 256;            
            const k20 = 1 / 256;
            const k21 = 4 / 256;
            const k22 = 6 / 256;
            const k23 = 4 / 256;
            const k24 = 1 / 256;            
            if (this.thread.y > 1 && this.thread.y < 480 - 2 && this.thread.x < 640 - 2 && this.thread.x > 1) {
                const a0 = frame[this.thread.y + 2][this.thread.x - 2];
                const a1 = frame[this.thread.y + 2][this.thread.x - 1];
                const a2 = frame[this.thread.y + 2][this.thread.x];
                const a3 = frame[this.thread.y + 2][this.thread.x + 1];
                const a4 = frame[this.thread.y + 2][this.thread.x + 2];                
                const a5 = frame[this.thread.y + 1][this.thread.x - 2];
                const a6 = frame[this.thread.y + 1][this.thread.x - 1];
                const a7 = frame[this.thread.y + 1][this.thread.x];
                const a8 = frame[this.thread.y + 1][this.thread.x + 1];
                const a9 = frame[this.thread.y + 1][this.thread.x + 2];                
                const a10 = frame[this.thread.y][this.thread.x - 2];
                const a11 = frame[this.thread.y][this.thread.x - 1];
                const a12 = frame[this.thread.y][this.thread.x];
                const a13 = frame[this.thread.y][this.thread.x + 1];
                const a14 = frame[this.thread.y][this.thread.x + 2];                
                const a15 = frame[this.thread.y - 1][this.thread.x - 2];
                const a16 = frame[this.thread.y - 1][this.thread.x - 1];
                const a17 = frame[this.thread.y - 1][this.thread.x];
                const a18 = frame[this.thread.y - 1][this.thread.x + 1];
                const a19 = frame[this.thread.y - 1][this.thread.x + 2];                
                const a20 = frame[this.thread.y - 2][this.thread.x - 2];
                const a21 = frame[this.thread.y - 2][this.thread.x - 1];
                const a22 = frame[this.thread.y - 2][this.thread.x];
                const a23 = frame[this.thread.y - 2][this.thread.x + 1];
                const a24 = frame[this.thread.y - 2][this.thread.x + 2];
                for (var i = 0; i < 3; i++) { // Compute the convolution for each of red [0], green [1] and blue [2]
                    col[i] = a0[i] * k0 + a1[i] * k1 + a2[i] * k2 + a3[i] * k3 + a4[i]* k4 
                                + a5[i] * k5 + a6[i] * k6 + a7[i] * k7 + a8[i] * k8 + a9[i] * k9
                                + a10[i] * k10 + a11[i] * k11 + a12[i] * k12 + a13[i] * k13 + a14[i] * k14
                                + a15[i] * k15 + a16[i] * k16 + a17[i] * k17 + a18[i] * k18 + a19[i] * k19
                                + a20[i] * k20 + a21[i] * k21 + a22[i] * k22 + a23[i] * k23 + a24[i] * k24;
                }
                this.color(col[0], col[1], col[2], pixel.a);
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
    
    // Kernel for performing edge detection
    const edgeDetectionKernel = gpu.createKernel(function (frame, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        var gx = [0, 0, 0];
        var gy = [0, 0, 0];
        if (isEnabled) {
            const x0 = 1;
            const x1 = 0;
            const x2 = -1;
            const x3 = 2;
            const x4 = 0;
            const x5 = -2;
            const x6 = 1;
            const x7 = 0;
            const x8 = -1;
            const y0 = 1;
            const y1 = 2;
            const y2 = 1;
            const y3 = 0;
            const y4 = 0;
            const y5 = 0;
            const y6 = -1;
            const y7 = -2;
            const y8 = -1;
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
                for (var i = 0; i < 3; i++) { // Get the horizontal derivative approximation for each colour component
                    gx[i] = a0[i] * x0 + a1[i] * x1 + a2[i] * x2 + a3[i] * x3 + a4[i]* x4 
                                + a5[i] * x5 + a6[i] * x6 + a7[i] * x7 + a8[i] * x8;
                }
                for (var i = 0; i < 3; i++) { // Get the vertical derivative approximation for each colour component
                    gy[i] = a0[i] * y0 + a1[i] * y1 + a2[i] * y2 + a3[i] * y3 + a4[i]* y4 
                                + a5[i] * y5 + a6[i] * y6 + a7[i] * y7 + a8[i] * y8;
                }
                // Compute the resultant gradient approximation 
                this.color(Math.sqrt(gx[0] * gx[0] + gy[0] * gy[0]), Math.sqrt(gx[1] * gx[1] + gy[1] * gy[1]),
                            Math.sqrt(gx[2] * gx[2] + gy[2] * gy[2]) , pixel.a);
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
            
    // Kernel for performing encryption on pixel colours
    const encryptKernel = gpu.createKernel(function (frame, key, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        var col = [0,0,0];
        if (isEnabled) {
            for (var i=0; i < 3; i++) { // Get the encrypted value for each colour component
                col[i] = Math.round(key[3 * i] * pixel.r * 255 + key[3 * i + 1] * pixel.g * 255 + key[3 * i + 2] * pixel.b * 255) % 256 / 255;
	        }
            this.color(col[0], col[1], col[2], pixel.a);
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
    
    // Kernel for performing decryption on pixel colours
    const decryptKernel = gpu.createKernel(function (frame, key, isEnabled) {
        const pixel = frame[this.thread.y][this.thread.x];
        var col = [0,0,0];
        if (isEnabled) {
            for (var i = 0; i < 3 ; i++) {// Get the decrypted value for each colour component
                col[i] = Math.floor(key[3 * i] * pixel.r * 255 + key[3 * i + 1]
                            * pixel.g * 255 + key[3 * i + 2] * pixel.b * 255) % 256 / 255;
	        }
            this.color(col[0], col[1], col[2], pixel.a);
        } else {
            this.color(pixel.r, pixel.g, pixel.b, pixel.a);
        }
    }, {
        pipeline: true,
        output: [640, 480],
        graphical: true,
        tactic: 'precision'
    });
    
    // Kernel for converting texture into displayable output
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
            // Start of pipeline
            const outputStage1 = horizontalFlipKernal(videoElement, horizontalFlip.checked);
            const outputStage2 = verticalFlipKernal(outputStage1, verticalFlip.checked);
            const outputStage3 = sharpenKernel(outputStage2, sharpen.checked);
            const outputStage4 = gaussianBlurKernel(outputStage3, gaussianBlur.checked);
            const outputStage5 = invertColourKernel(outputStage4, invertColours.checked);
            const outputStage6 = grayscaleColourKernal(outputStage5, grayscaleColours.checked);
            const outputStage7 = edgeDetectionKernel(outputStage6, edgeDetection.checked);
            const outputStage8 = encryptKernel(outputStage7, getKey(0), encrypt.checked);
            const outputStage9 = decryptKernel(outputStage8, getKey(1), decrypt.checked);
            textureToImageKernel(outputStage9); // End of pipeline
            
            var t1 = performance.now();
            var delta = t1 - t0;
            timeTaken.innerHTML = delta.toFixed(3);
        } else {
            // Prevents users from running more than one kernel at one time if GPU is disabled
            switch (getRadioValue()) {
                case "horizontal-flip":
                    var t0 = performance.now();
                    horizontalFlipKernal(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(3);
                    break;
                case "vertical-flip":
                    var t0 = performance.now();
                    verticalFlipKernal(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(3);
                    break;
                case "sharpen":
                    var t0 = performance.now();
                    sharpenKernel(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(3);
                    break;
                case "gaussian-blur":
                    var t0 = performance.now();
                    gaussianBlurKernel(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(3);
                    break;
                case "invert-colours":
                    var t0 = performance.now();
                    invertColourKernel(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(3);
                    break;
                case "grayscale-colours":
                    var t0 = performance.now();
                    grayscaleColourKernal(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(3);
                    break;
                case "edge-detection":
                    var t0 = performance.now();
                    edgeDetectionKernel(videoElement, true);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(3);
                    break;
                case "no-filter":
                    var t0 = performance.now();
                    textureToImageKernel(videoElement);
                    var t1 = performance.now();
                    var delta = t1 - t0;
                    timeTaken.innerHTML = delta.toFixed(3);
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

// Toggle UI between GPU mode and CPU mode
function toggleDiv(){
    if (enableGpu.checked == true){
        document.getElementById("gpu-controls").style.display = "block";
        document.getElementById("cpu-controls").style.display = "none";
    } else {
        document.getElementById("gpu-controls").style.display = "none";
        document.getElementById("cpu-controls").style.display = "block";
    }
}

// Get the current selected kernel from user's input
function getRadioValue() {
    var elements = document.getElementsByName('filter');    
    for(i = 0; i < elements.length; i++) { 
        if(elements[i].checked) {
            return elements[i].value;
        }
    } 
}

// Get the key value used by Hill Cipher from user's input
function getKey(type) {
    var elements;
    if (type == 0) {
         elements = document.getElementsByName('encryption-key');   
    } else {
         elements = document.getElementsByName('decryption-key');   
    }
    var key;     
    for(i = 0; i < elements.length; i++) { 
        if(elements[i].checked) {
            key = elements[i].value;
        }
    }
    
    return getKeyValue(key);
}

function getKeyValue(key) {
    switch (key) {
        case "ekey1":
            keyValue = [6, 24, 1, 13, 16, 10, 20, 17, 15];
            break;
        case "ekey2":
            keyValue = [14, 51, 1, 22, 75, 12, 3, 42, 12];
            break;
        case "ekey3":
            keyValue = [212, 234, 111, 196, 67, 130, 93, 219, 8];
            break;
        case "dkey1":
            keyValue = [118, 113, 224, 173, 118, 217, 5, 74, 104];
            break;
        case "dkey2":
            keyValue = [212, 234, 111, 196,	67, 130, 93, 219, 8];
            break;
        case "dkey3":
            keyValue = [14,	51,	1, 22, 75, 12, 3, 42, 12];
            break;
    }
    return keyValue;
}