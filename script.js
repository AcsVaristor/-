const video = document.getElementById('video');
const label = document.getElementById('label');

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('models'),
  faceapi.nets.faceRecognitionNet.loadFromUri('models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('models')
]).then(startVideo);

function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: {} })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error(err));
}

video.addEventListener('play', async () => {
  const labeledDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const text = result.toString();
      label.textContent = `結果: ${text}`;
      const drawBox = new faceapi.draw.DrawBox(box, { label: text });
      drawBox.draw(canvas);
    });
  }, 1000);
});

function loadLabeledImages() {
  const labels = ['yuki', 'haruka']; // known_faces フォルダの画像名（拡張子なし）
  return Promise.all(
    labels.map(async label => {
      const imgUrl = `known_faces/${label}.png`;
      const img = await faceapi.fetchImage(imgUrl);
      const detections = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
      return new faceapi.LabeledFaceDescriptors(label, [detections.descriptor]);
    })
  );
}
