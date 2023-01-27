import { build_hierarchy_wasm, display_labels_wasm, cut_hierarchy_wasm, Hierarchy } from 'hierarchy_labellisation';
import { fromArrayBuffer, TypedArray } from 'geotiff';

let hierarchy: Hierarchy | null = null;
let tiff: Awaited<ReturnType<typeof readTiff>> | null = null;

function setupFileInput() {
    const fileSelector = document.getElementById('file-selector') as HTMLInputElement;
    fileSelector.addEventListener('change', (_) => {
        const files = fileSelector.files!;

        for (let i = 0; i < files.length; i++) {
            const file = files.item(i)!;

            console.log(`Loading file (${file.name})...`);

            const reader = new FileReader();
            reader.onload = (_) => {
                const arrayBuffer = reader.result as ArrayBuffer;

                processTiff(arrayBuffer);
            }
            reader.readAsArrayBuffer(file);
        }
    });
}

function setupSlider() {
    const slider = document.getElementById('slider') as HTMLInputElement;

    let working = false;
    slider.addEventListener('input', async (_) => {
        const value = slider.valueAsNumber;
        if (working) {
            return;
        }
        working = true;
        await handleSlider(value);

        requestAnimationFrame(() => {
            working = false;
        });
    });
}

async function readTiff(buffer: ArrayBuffer) {
    const tiff = await fromArrayBuffer(buffer);
    const image = await tiff.getImage();

    const width = image.getWidth();
    const height = image.getHeight();
    const channels = image.getSamplesPerPixel();

    const bytesPerValue = image.getBytesPerPixel() / channels;
    if (bytesPerValue !== 1) {
        throw new Error('Only 8-bit images are supported');
    }

    const data = await image.readRasters() as TypedArray[];

    // Merge channels into a single array
    const merged = new Uint8Array(width * height * channels);
    data.forEach((channel, i) => {
        merged.set(channel, i * width * height);
    });

    return {
        width,
        height,
        channels,
        data: merged,
    }
}

async function processTiff(buffer: ArrayBuffer) {
    tiff = await readTiff(buffer);

    console.log(`Loaded image (${tiff.width}x${tiff.height}x${tiff.channels}).`);
    console.log('Building hierarchy...')
    const clusterCount = Math.round(tiff.width * tiff.height / 200);
    hierarchy = build_hierarchy_wasm(tiff.data, tiff.width, tiff.height, tiff.channels, clusterCount);

    console.log(hierarchy);

    console.log('Cutting hierarchy...');
    const labels = cut_hierarchy_wasm(hierarchy, 0);

    console.log('Displaying labels...');
    const bitmapResult = display_labels_wasm(tiff.data, tiff.width, tiff.height, labels);

    const uint8ClampedArray = new Uint8ClampedArray(bitmapResult);
    const imageData = new ImageData(uint8ClampedArray, tiff.width, tiff.height);
    const imageBitmap = await createImageBitmap(imageData);

    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    canvas.width = tiff.width;
    canvas.height = tiff.height;
    ctx.drawImage(imageBitmap, 0, 0);
}

async function handleSlider(value: number) {
    if (hierarchy === null || tiff === null) {
        return;
    }

    const maxValue = Math.log2(hierarchy.max_level);
    const logValue = value * maxValue;
    const level = Math.pow(2, logValue);

    console.log('Cutting hierarchy...');
    const labels = cut_hierarchy_wasm(hierarchy, level);

    console.log('Displaying labels...');
    const bitmapResult = display_labels_wasm(tiff.data, tiff.width, tiff.height, labels);

    console.log('Rendering canvas...');
    const uint8ClampedArray = new Uint8ClampedArray(bitmapResult);
    const imageData = new ImageData(uint8ClampedArray, tiff.width, tiff.height);
    const imageBitmap = await createImageBitmap(imageData);

    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    canvas.width = tiff.width;
    canvas.height = tiff.height;
    ctx.drawImage(imageBitmap, 0, 0);

    console.log('Done.');
}

setupFileInput();
setupSlider();
