# Hierarchy Labellisation

This repository contains the code for an experimental hierarchical labellisation tool. It is developped in Rust and compiles to WebAssembly.

![Preview](https://i.imgur.com/Vhp9djB.jpg)

It was developped as an end-of-study project at [EPITA](https://www.epita.fr/). The project was supervised by [Nicolas David](https://github.com/ndavid) at IGN.

## Installation

### Prerequisites

- Rust 1.67+
- wasm-pack 0.10.3+
- Node.js + Yarn or NPM (for the example)

### Build

```bash
wasm-pack build
```

This will generate a `pkg` folder containing the compiled WebAssembly module. It can be imported in any Javascript project using a bundler such as Webpack or Vite.

The file `/pkg/hierarchy_labellisation.d.ts` contains the type definitions and exported functions. You can use them to interact with the module.

## Usage

You can find a working example in the `example` folder. The example is a simple vanilla Typescript project bundled with Vite. It allows you to load a TIFF image and compute its hierarchical segmentation. You can then use the slider to change the segmentation level.

To run it, simply run the following commands:

```bash
cd example
yarn install # You can also use `npm install`
yarn dev # You can also use `npm run dev`
```

The example will be available at http://127.0.0.1:5173/.
