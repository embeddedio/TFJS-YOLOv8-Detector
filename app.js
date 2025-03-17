const categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
    const numClasses = categories.length;
    const modelName = "yolov8n";
    let loadedModel = null;
    let inputDimensions = [1, 0, 0, 3];
    let currentStream = null;
    let confidenceThreshold = 0.5;

    function updateStatus(loading, progress) {
      document.getElementById('status').style.display = loading ? 'block' : 'none';
      document.getElementById('status').innerText = `Loading model... ${(progress * 100).toFixed(2)}%`;
    }

    function prepareInput(source, modelWidth, modelHeight) {
      let originalWidth, originalHeight;
      if (source === document.getElementById('media')) {
        originalWidth = source.naturalWidth;
        originalHeight = source.naturalHeight;
      } else {
        originalWidth = source.videoWidth;
        originalHeight = source.videoHeight;
      }
      const xScale = originalWidth / modelWidth;
      const yScale = originalHeight / modelHeight;
      const img = tf.browser.fromPixels(source);
      const resized = tf.image.resizeBilinear(img, [modelWidth, modelHeight]);
      const normalized = resized.div(255.0).expandDims(0);
      return [normalized, xScale, yScale, originalWidth, originalHeight];
    }

    async function processImage(source, callback = () => {}) {
      const [modelWidth, modelHeight] = inputDimensions.slice(1, 3);
      tf.engine().startScope();
      const [input, xScale, yScale, originalWidth, originalHeight] = prepareInput(source, modelWidth, modelHeight);
      const results = loadedModel.execute(input);
      const transposed = results.transpose([0, 2, 1]);
      const boundingBoxes = tf.tidy(() => {
        const widths = transposed.slice([0, 0, 2], [-1, -1, 1]);
        const heights = transposed.slice([0, 0, 3], [-1, -1, 1]);
        const x1 = tf.sub(transposed.slice([0, 0, 0], [-1, -1, 1]), tf.div(widths, 2));
        const y1 = tf.sub(transposed.slice([0, 0, 1], [-1, -1, 1]), tf.div(heights, 2));
        return tf.concat([y1, x1, tf.add(y1, heights), tf.add(x1, widths)], 2).squeeze();
      });
      const [scores, classes] = tf.tidy(() => {
        const rawScores = transposed.slice([0, 0, 4], [-1, -1, numClasses]).squeeze(0);
        return [rawScores.max(1), rawScores.argMax(1)];
      });
      const filteredIndices = await tf.image.nonMaxSuppressionAsync(boundingBoxes, scores, 500, 0.45, 0.2);
      const boundingBoxData = boundingBoxes.gather(filteredIndices, 0).dataSync();
      const scoreData = scores.gather(filteredIndices, 0).dataSync();
      const classData = classes.gather(filteredIndices, 0).dataSync();
      drawBoxes(document.getElementById('overlay'), boundingBoxData, scoreData, classData, [xScale, yScale], originalWidth, originalHeight, source);
      tf.dispose([results, transposed, boundingBoxes, scores, classes, filteredIndices]);
      callback();
      tf.engine().endScope();
    }

    function drawBoxes(canvasRef, boxes, scores, classes, scales, originalWidth, originalHeight, source) {
      const ctx = canvasRef.getContext("2d");
      const displayedWidth = source.clientWidth;
      const displayedHeight = source.clientHeight;
      const xScaleFactor = displayedWidth / originalWidth;
      const yScaleFactor = displayedHeight / originalHeight;
      ctx.canvas.width = displayedWidth;
      ctx.canvas.height = displayedHeight;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      const colorPalette = createColorPalette();
      const fontSize = `${Math.max(Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40), 14)}px Arial`;
      ctx.font = fontSize;
      ctx.textBaseline = "top";
      for (let i = 0; i < scores.length; ++i) {
        if (scores[i] < confidenceThreshold) continue;
        const className = categories[classes[i]];
        const color = colorPalette[classes[i] % colorPalette.length];
        const scoreText = (scores[i] * 100).toFixed(1);
        const [y1, x1, y2, x2] = boxes.slice(i * 4, (i + 1) * 4);
        const scaledX1 = (x1 / 640) * originalWidth * xScaleFactor;
        const scaledX2 = (x2 / 640) * originalWidth * xScaleFactor;
        const scaledY1 = (y1 / 640) * originalHeight * yScaleFactor;
        const scaledY2 = (y2 / 640) * originalHeight * yScaleFactor;
        const width = scaledX2 - scaledX1;
        const height = scaledY2 - scaledY1;
        ctx.fillStyle = rgbaString(color, 0.2);
        ctx.fillRect(scaledX1, scaledY1, width, height);
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5);
        ctx.strokeRect(scaledX1, scaledY1, width, height);
        const textWidth = ctx.measureText(`${className} - ${scoreText}%`).width;
        const textHeight = parseInt(fontSize, 10);
        const textY = scaledY1 - textHeight - ctx.lineWidth;
        ctx.fillStyle = color;
        ctx.fillRect(scaledX1 - 1, textY < 0 ? 0 : textY, textWidth + ctx.lineWidth, textHeight + ctx.lineWidth);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(`${className} - ${scoreText}%`, scaledX1 - 1, textY < 0 ? 0 : textY);
      }
    }

    function createColorPalette() {
      return [
        "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
        "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
        "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
        "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
      ];
    }

    function rgbaString(hex, alpha) {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? `rgba(${[parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)].join(", ")}, ${alpha})` : null;
    }

    function processVideo(vidSource) {
      const frameProcessor = async () => {
        if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
          const ctx = document.getElementById('overlay').getContext("2d");
          ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
          return;
        }
        processImage(vidSource, () => {
          requestAnimationFrame(frameProcessor);
        });
      };
      frameProcessor();
    }

    function startCamera(videoRef) {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: "environment" }
        }).then((stream) => {
          videoRef.srcObject = stream;
        });
      } else {
        alert("Unable to access the webcam.");
      }
    }

    function stopCamera(videoRef) {
      if (videoRef.srcObject) {
        videoRef.srcObject.getTracks().forEach(track => track.stop());
        videoRef.srcObject = null;
      } else {
        alert("No active webcam stream.");
      }
    }

    function closeMedia(elementId) {
      const url = document.getElementById(elementId).src;
      document.getElementById(elementId).src = "#";
      URL.revokeObjectURL(url);
      currentStream = null;
      document.getElementById(elementId).style.display = "none";
    }

    function handleImageButton() {
      if (!currentStream) {
        document.getElementById('file-input-image').click();
      } else if (currentStream === "image") {
        closeMedia("media");
      } else {
        alert(`Cannot handle multiple streams.\nCurrent stream: ${currentStream}`);
      }
    }

    function handleFileInputChange(event) {
      const file = event.target.files[0];
      if (file) {
        const url = URL.createObjectURL(file);
        document.getElementById('media').src = url;
        document.getElementById('media').style.display = "block";
        currentStream = "image";
      }
    }

    function handleVideoButton() {
      if (!currentStream || currentStream === "image") {
        document.getElementById('file-input-video').click();
      } else if (currentStream === "video") {
        closeMedia("video");
      } else {
        alert(`Cannot handle multiple streams.\nCurrent stream: ${currentStream}`);
      }
    }

    function handleVideoFileChange(event) {
      if (currentStream === "image") closeMedia("media");
      const file = event.target.files[0];
      if (file) {
        const url = URL.createObjectURL(file);
        document.getElementById('video').src = url;
        document.getElementById('video').addEventListener("ended", () => closeMedia("video"));
        document.getElementById('video').style.display = "block";
        currentStream = "video";
      }
    }

    function handleWebcamButton() {
      if (!currentStream || currentStream === "image") {
        if (currentStream === "image") closeMedia("media");
        startCamera(document.getElementById('webcam'));
        document.getElementById('webcam').style.display = "block";
        currentStream = "webcam";
      } else if (currentStream === "webcam") {
        stopCamera(document.getElementById('webcam'));
        document.getElementById('webcam').style.display = "none";
        currentStream = null;
      } else {
        alert(`Cannot handle multiple streams.\nCurrent stream: ${currentStream}`);
      }
    }

    document.getElementById('media').addEventListener('load', () => processImage(document.getElementById('media')));
    document.getElementById('webcam').addEventListener('play', () => processVideo(document.getElementById('webcam')));
    document.getElementById('video').addEventListener('play', () => processVideo(document.getElementById('video')));

    document.getElementById('open-image').addEventListener('click', handleImageButton);
    document.getElementById('file-input-image').addEventListener('change', handleFileInputChange);

    document.getElementById('open-video').addEventListener('click', handleVideoButton);
    document.getElementById('file-input-video').addEventListener('change', handleVideoFileChange);

    document.getElementById('open-camera').addEventListener('click', handleWebcamButton);

    document.getElementById('confidence').addEventListener('input', (e) => {
      confidenceThreshold = parseFloat(e.target.value);
      document.getElementById('confidence-value').innerText = confidenceThreshold.toFixed(2);
    });

    tf.ready().then(async () => {
      const model = await tf.loadGraphModel('model/model.json', {
        onProgress: (progress) => updateStatus(true, progress)
      });
      const dummyInput = tf.ones(model.inputs[0].shape);
      const warmupResult = model.execute(dummyInput);
      updateStatus(false, 1);
      loadedModel = model;
      inputDimensions = model.inputs[0].shape;
      tf.dispose([warmupResult, dummyInput]);
    });