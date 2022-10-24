import { add, convert_to_png, slic_from_js } from 'hierarchy_labellisation';
import { fromArrayBuffer, TypedArray } from 'geotiff';

console.log(add(1, 2));

function setupFileInput() {
    const fileSelector = document.getElementById('file-selector') as HTMLInputElement;
    fileSelector.addEventListener('change', (_) => {
        const files = fileSelector.files!;

        for (let i = 0; i < files.length; i++) {
            const file = files.item(i)!;

            console.log(file.name);

            const reader = new FileReader();
            reader.onload = (_) => {
                const arrayBuffer = reader.result as ArrayBuffer;

                processTiff(arrayBuffer);
            }
            reader.readAsArrayBuffer(file);
        }
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
    const tiff = await readTiff(buffer);

    // const result = convert_to_png(tiff.data, tiff.width, tiff.height, tiff.channels);
    const result = slic_from_js(tiff.data, tiff.width, tiff.height, tiff.channels, 2000, 10);


    const blob = new Blob([result], { type: 'image/png' });
    const url = URL.createObjectURL(blob);

    const img = document.createElement('img');
    img.src = url;
    img.width = 600;
    const app = document.getElementById('app')!;
    app.appendChild(img);
}

setupFileInput();
