async function handleImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    document.getElementById('status').textContent = "Status: Loading...";

    const imageBitmap = await createImageBitmap(file);
    const canvas = document.createElement('canvas');
    canvas.width = 96;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageBitmap, 0, 0, 96, 96);

    const imageData = ctx.getImageData(0, 0, 96, 96).data;

    const input = preprocessImage(imageData);

    await runModel(input);
}

function preprocessImage(imageData) {
    const floatArray = new Float32Array(1 * 3 * 96 * 96);

    let rIndex = 0 * 96 * 96;
    let gIndex = 1 * 96 * 96;
    let bIndex = 2 * 96 * 96;

    for (let i = 0, pxIdx = 0; i < imageData.length; i += 4, pxIdx++) {
        let r = imageData[i] / 255.0;
        let g = imageData[i + 1] / 255.0;
        let b = imageData[i + 2] / 255.0;

        r = (r - 0.5) / 0.5;
        g = (g - 0.5) / 0.5;
        b = (b - 0.5) / 0.5;

        floatArray[rIndex + pxIdx] = r;
        floatArray[gIndex + pxIdx] = g;
        floatArray[bIndex + pxIdx] = b;
    }

    return new ort.Tensor('float32', floatArray, [1, 3, 96, 96]);
}

async function runModel(inputTensor) {
    document.getElementById('status').textContent = "Status: Running...";

    const session = await ort.InferenceSession.create('./cifar10_resnext50.onnx');
    const feeds = { input: inputTensor };
    const outputData = await session.run(feeds);
    
    let output = outputData.output.data;
    
    output = softmax(output);

    showPredictions(output);
}

function softmax(arr) {
    const max = Math.max(...arr); 
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sum);
}

function showPredictions(output) {
    document.getElementById('status').textContent = "Status: Finished!";
    
    const labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ];

    let html = `<h2>Predictions</h2><table>`;
    for (let i = 0; i < labels.length; i++) {
        html += `<tr><td>${labels[i]}</td><td>${(output[i] * 100).toFixed(2)}%</td></tr>`;
    }
    html += `</table>`;

    document.getElementById('predictions').innerHTML = html;
}
