var ndarray = require('C:/Users/shrey/node_modules/ndarray/ndarray.js');
var ops = require('C:/Users/shrey/node_modules/ndarray-ops/ndarray-ops.js');
var unpack = require('C:/Users/shrey/node_modules/ndarray-unpack/unpack.js');
var flatten = _.flatten;

var width = 227;
var height = 227;

const model = new KerasJS.Model({
    filepaths: {
        model: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1.json',
        weights: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1_weights.buf',
        metadata: 'https://transcranial.github.io/keras-js-demos-data/squeezenet_v1.1/squeezenet_v1.1_metadata.json'
    },
    gpu: true
});

$(document).ready(function() {
    model.ready().then(function() {
        console.log('Model is ready.');
        $('.loading-indicator').hide();
        setupwebcam();
    });

    function setupwebcam() {

        // Grab elements, create settings, etc.
        var canvas = document.getElementById("canvas"),
            context = canvas.getContext("2d"),
            // we don't need to append the video to the document
            video = document.createElement("video"),
            videoObj =
            navigator.getUserMedia || navigator.mozGetUserMedia ? // our browser is up to date with specs ?
            {
                video: {
                    width: {
                        min: width,
                        max: width
                    },
                    height: {
                        min: height,
                        max: height
                    },
                    require: ['width', 'height']
                }
            } : {
                video: {
                    mandatory: {
                        minWidth: width,
                        minHeight: height,
                        maxWidth: width,
                        maxHeight: height
                    }
                }
            };


        errBack = function(error) {
            console.log("Video capture error: ", error.code);
        };
        // create a crop object that will be calculated on load of the video
        var crop;
        // create a variable that will enable us to stop the loop.
        var raf;

        navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
        // Put video listeners into place
        navigator.getUserMedia(videoObj, function(stream) {
            video.src = URL.createObjectURL(stream);
            video.onplaying = function() {
                var croppedWidth = (Math.min(video.videoHeight, canvas.height) / Math.max(video.videoHeight, canvas.height)) * Math.min(video.videoWidth, canvas.width),
                    croppedX = (video.videoWidth - croppedWidth) / 2;
                crop = {
                    w: croppedWidth,
                    h: video.videoHeight,
                    x: croppedX,
                    y: 0
                };
                // call our loop only when the video is playing
                raf = requestAnimationFrame(loop);
            };
            video.onpause = function() {
                // stop the loop
                cancelAnimationFrame(raf);
            }
            video.play();
        }, errBack);

        function loop() {
            context.drawImage(video, crop.x, crop.y, crop.w, crop.h, 0, 0, canvas.width, canvas.height);
            mainWebcamLoop().then(function(output) {
                var predictions = imagenetClassesTopK(output['loss']);
                $('.results-container').text('');
                var results_str = '';
                for (var p in predictions) {
                    results_str += predictions[p].probability.toFixed(2) + ' ' + predictions[p].name + '\n';
                }
                $('.results-container').text(results_str);

                let results = []
                for (let [name, layer] of model.modelLayersMap.entries()) {
                    if (name === 'input') continue
                    const layerClass = layer.layerClass || ''
                    let images = []
                    if (layer.result && layer.result.tensor.shape.length === 3) {
                        images = unroll3Dtensor(layer.result.tensor)
                    } else if (layer.result && layer.result.tensor.shape.length === 2) {
                        images = [image2Dtensor(layer.result.tensor)]
                    } else if (layer.result && layer.result.tensor.shape.length === 1) {
                        images = [image1Dtensor(layer.result.tensor)]
                    }
                    results.push({
                        name,
                        layerClass,
                        images
                    })
                }
                for (let x of results) {
                    if (x.name == 'conv10') console.log(x.images[0].width);
                }
                raf = requestAnimationFrame(loop);
            });

        }
        // now that our video is drawn correctly, we can do...
        context.translate(canvas.width, 0);
        context.scale(-1, 1);
    }

}, false);

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

function mainWebcamLoop() {
    var canvas = document.getElementById("canvas");
    var context = canvas.getContext("2d");

    const imageData = context.getImageData(0, 0, context.canvas.width, context.canvas.height);
    const {
        data,
        aasa,
        aa
    } = imageData;

    // data processing
    // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
    var dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    var dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3]);
    ops.subseq(dataTensor.pick(null, null, 2), 103.939)
    ops.subseq(dataTensor.pick(null, null, 1), 116.779)
    ops.subseq(dataTensor.pick(null, null, 0), 123.68)
    ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 2))
    ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
    ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 0))
    var inputData = {
        input_1: dataProcessedTensor.data
    }

    return model.predict(inputData);

}

function tensorMinMax(tensor) {
    let min = Infinity
    let max = -Infinity
    for (let i = 0, len = tensor.data.length; i < len; i++) {
        if (tensor.data[i] < min) min = tensor.data[i]
        if (tensor.data[i] > max) max = tensor.data[i]
    }
    return {
        min,
        max
    }
}

function image1Dtensor(tensor) {
    const {
        min,
        max
    } = tensorMinMax(tensor)
    let imageData = new Uint8ClampedArray(tensor.size * 4)
    for (let i = 0, len = imageData.length; i < len; i += 4) {
        imageData[i + 3] = 255 * (tensor.data[i / 4] - min) / (max - min)
    }
    return new ImageData(imageData, tensor.shape[0], 1)
}

function image2Dtensor(tensor) {
    const {
        min,
        max
    } = tensorMinMax(tensor)
    let imageData = new Uint8ClampedArray(tensor.size * 4)
    for (let i = 0, len = imageData.length; i < len; i += 4) {
        imageData[i + 3] = 255 * (tensor.data[i / 4] - min) / (max - min)
    }
    return new ImageData(imageData, tensor.shape[0], tensor.shape[1])
}

function unroll3Dtensor(tensor) {
    const {
        min,
        max
    } = tensorMinMax(tensor)
    let shape = tensor.shape.slice()
    let unrolled = []
    for (let k = 0, channels = shape[2]; k < channels; k++) {
        const channelData = flatten(unpack(tensor.pick(null, null, k)))
        unrolled.push(channelData)
    }

    return unrolled.map(channelData => {
        let imageData = new Uint8ClampedArray(channelData.length * 4)
        for (let i = 0, len = channelData.length; i < len; i++) {
            imageData[i * 4] = 0
            imageData[i * 4 + 1] = 0
            imageData[i * 4 + 2] = 0
            imageData[i * 4 + 3] = 255 * (channelData[i] - min) / (max - min)
        }
        return new ImageData(imageData, shape[0], shape[1])
    })
}

function imagenetClassesTopK(classProbabilities, k = 5) {
  const probs = _.isTypedArray(classProbabilities) ? Array.prototype.slice.call(classProbabilities) : classProbabilities

  const sorted = _.reverse(_.sortBy(probs.map((prob, index) => [prob, index]), probIndex => probIndex[0]))

  const topK = _.take(sorted, k).map(probIndex => {
    const iClass = imagenet_classes[probIndex[1]]
    return { id: probIndex[1], name: iClass, probability: probIndex[0] }
  })
  return topK
}