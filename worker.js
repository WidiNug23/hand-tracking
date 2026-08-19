// Memuat MediaPipe Tasks Vision langsung di dalam Web Worker
importScripts("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js");

let handLandmarker = null;

async function initModel() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "IMAGE",
        numHands: 2
    });
    postMessage({ type: "READY" });
}

initModel();

// Menerima frame gambar dari index.html dan memprosesnya di background
onmessage = async (event) => {
    if (!handLandmarker) return;
    const { imageBitmap, timestamp } = event.data;
    
    try {
        const results = handLandmarker.detect(imageBitmap);
        // Tutup imageBitmap untuk mencegah kebocoran memori (memory leak)
        imageBitmap.close();
        postMessage({ type: "RESULTS", results, timestamp });
    } catch (error) {
        imageBitmap.close();
        // Abaikan error sesaat jika frame terlewat
    }
};