{
  description = "Python development template";

  inputs = {
    utils.url = "github:numtide/flake-utils";
    nixos.url = "github:mordragt/nixos";
  };

  outputs = {
    self,
    nixpkgs,
    nixos,
    utils,
    ...
  }:
    utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          nixos.overlays.default
        ];
      };
      python = pkgs.intel-python;
      pythonPkgs = python.pkgs;
      dependencies = p:
        with p; [
          # Python dependencies
          numpy
          pandas
          praw
          moviepy
          gradio
          transformers
          parler-tts
          torch
          torchaudio
          torchvision
          accelerate # for bark
          optimum-intel
          # optimum

          ipex
          bitsandbytes
          python-dotenv
          outlines # structured text generation
          pydantic
          # gguf
          # spacy-transformers
          # pip # for download spacy
          # scipy
          # sentencepiece # for speecht5
          # datasets # for speecht5
          pyproject-parser # for validation
          consolekit # needed for pyporject-parser
          triton-xpu
          huggingface-hub
          inquirerpy # for huggingface-hub cli

          # pybind11 # for onnx ?
        ];
      # ++ optimum.optional-dependencies.exporters
      # ++ optimum-intel.optional-dependencies.openvino;
    in {
      packages.default = pythonPkgs.buildPythonPackage {
        pname = "rizz";
        version = "0.1.0";
        format = "pyproject";

        src = ./.;

        build-system = [pythonPkgs.hatchling];

        dependencies = dependencies pythonPkgs;

        buildInputs = with pkgs; [
          espeak
          imagemagick
          intel-dpcpp.clang
          level-zero
        ];
      };
      devShells.default = pkgs.mkShell {
        inputsFrom = [self.packages.${system}.default];

        buildInputs = [
          (python.withPackages dependencies)
          pkgs.intel-dpcpp.clang
          pkgs.level-zero
        ];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          espeak
          imagemagick
        ]);
      };
    });
}
