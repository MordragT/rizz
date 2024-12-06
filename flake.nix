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
      pythonPkgs = pkgs.intel-python.pkgs;
    in {
      packages = rec {
        moviepy = pythonPkgs.buildPythonPackage rec {
          pname = "moviepy";
          version = "2.1.1";
          format = "pyproject";

          src = pkgs.fetchFromGitHub {
            owner = "Zulko";
            repo = "moviepy";
            rev = "refs/tags/v${version}";
            hash = "sha256-skfQbbWTXyTZjpehAcUdbyAPjZ5ZJoo69DAVWgCMF+k=";
          };

          postPatch = ''
            substituteInPlace pyproject.toml \
              --replace-fail "pillow>=9.2.0,<11.0" "pillow>=9.2.0,<12.0"
          '';

          build-system = [pythonPkgs.setuptools];

          dependencies = with pythonPkgs; [
            decorator
            imageio
            imageio-ffmpeg
            numpy
            proglog
            python-dotenv
            pillow
          ];

          pythonImportsCheck = ["moviepy"];
        };
        argbind = pythonPkgs.buildPythonPackage {
          pname = "argbind";
          version = "unstable-2024-05-24";
          format = "pyproject";

          src = pkgs.fetchFromGitHub {
            owner = "pseeth";
            repo = "argbind";
            rev = "e3e0b8d2d906e2b99879be7b726353498f29012b";
            hash = "sha256-3Iulc55GnMLqA/0HZJ3BG59xXd7Rg7VMpLWGhNmH8JM=";
          };

          build-system = [pythonPkgs.setuptools];

          dependencies = with pythonPkgs; [
            pyyaml
            docstring-parser
          ];

          pythonImportsCheck = ["argbind"];
        };
        randomname = pythonPkgs.buildPythonPackage {
          pname = "randomname";
          version = "0.2.1";
          format = "pyproject";

          src = pkgs.fetchFromGitHub {
            owner = "beasteers";
            repo = "randomname";
            rev = "14b419a536ce1c6a9a2220ac156a399c05720762";
            hash = "sha256-h1K9CaO95g5CMup6vRjLbQSU7TRmdeWOjme2oTc/2Ic=";
          };

          build-system = [pythonPkgs.setuptools];

          dependencies = with pythonPkgs; [
            fire
          ];

          pythonImportsCheck = ["randomname"];
        };
        # pyloudnorm = pythonPkgs.buildPythonPackage rec {
        #   pname = "pyloudnorm";
        #   version = "0.1.1";
        #   format = "pyproject";

        #   src = pkgs.fetchFromGitHub {
        #     owner = "csteinmetz1";
        #     repo = "pyloudnorm";
        #     rev = "v${version}";
        #     hash = "sha256-eIJrN/UU1oCnBJkNzUJXNykNq7tsUpKH4GZtoU4wKUk=";
        #   };

        #   build-system = [pythonPkgs.setuptools];

        #   dependencies = with pythonPkgs; [
        #     scipy
        #     numpy
        #   ];

        #   pythonImportsCheck = ["pyloudnorm"];
        # };
        descript-audiotools = pythonPkgs.buildPythonPackage rec {
          pname = "descript-audiotools";
          version = "0.7.4";
          format = "pyproject";

          src = pkgs.fetchFromGitHub {
            owner = "descriptinc";
            repo = "audiotools";
            rev = version;
            hash = "sha256-mDReVnVgxb+qcTosUSNG3jp6QhaIWdcddyfK4xuyxCc=";
          };

          build-system = [pythonPkgs.setuptools];

          dependencies = with pythonPkgs; [
            argbind
            numpy
            soundfile
            # pyloudnorm
            importlib-resources
            scipy
            torch
            julius
            torchaudio
            ffmpy
            ipython
            rich
            matplotlib
            librosa
            # "pystoi",
            # "torch_stoi",
            flatten-dict
            markdown2
            randomname
            protobuf
            tensorboard
            tqdm
          ];

          pythonRemoveDeps = [
            "pyloudnorm"
            "pystoi"
            "torch-stoi"
          ];

          pythonRelaxDeps = [
            "protobuf"
          ];

          pythonImportsCheck = ["audiotools"];
        };
        descript-audio-codec = pythonPkgs.buildPythonPackage rec {
          pname = "descript-audio-codec";
          version = "1.0.0";
          format = "pyproject";

          src = pkgs.fetchFromGitHub {
            owner = "descriptinc";
            repo = "descript-audio-codec";
            rev = version;
            hash = "sha256-cABV+wyon211I2Gvhu0hn+Y1D/RiQ6pjxU7qYMN71BU=";
          };

          build-system = [pythonPkgs.setuptools];

          dependencies = with pythonPkgs; [
            argbind
            descript-audiotools
            einops
            numpy
            torch
            torchaudio
            tqdm
            tensorboard
            numba
            jupyterlab
          ];

          pythonImportsCheck = ["dac"];
        };
        parler-tts = pythonPkgs.buildPythonPackage {
          pname = "parler-tts";
          version = "unstable-2024-12-02";
          format = "pyproject";

          src = pkgs.fetchFromGitHub {
            owner = "huggingface";
            repo = "parler-tts";
            rev = "77d10df6d230fbdedcf92fc561182701ae6710ca";
            hash = "sha256-XZJ9AmQKHp+kILkSZlHUxEdIG1xXyv5NqhCPo5EN3UU=";
          };

          build-system = [pythonPkgs.setuptools];

          dependencies = with pythonPkgs; [
            transformers
            torch
            sentencepiece
            descript-audiotools
            descript-audio-codec
          ];

          pythonRelaxDeps = [
            "transformers"
          ];

          pythonImportsCheck = ["parler_tts"];
        };
        default = pythonPkgs.buildPythonPackage {
          pname = "rizz";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          build-system = [pythonPkgs.hatchling];

          dependencies = with pythonPkgs; [
            # Python dependencies
            numpy
            praw
            moviepy
            gradio
            transformers
            parler-tts
            torch
            torchaudio
            ipex
            python-dotenv
            # optimum
            # gguf
            # accelerate
            # spacy-transformers
            # pip # for download spacy
            # scipy
            # sentencepiece # for speecht5
            # datasets # for speecht5
          ];

          buildInputs = with pkgs; [
            espeak
            imagemagick
            # sentencepiece
          ];
        };
      };
      devShells.default = pkgs.mkShell {
        inputsFrom = [self.packages.${system}.default];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          espeak
          imagemagick
          # sentencepiece
        ]);
      };
    });
}
