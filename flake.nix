{
  description = "Development shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
  };

outputs = { self, nixpkgs }: 
let
  system = "x86_64-linux";
  pkgs = import nixpkgs {
    system = system;
  };
  segmentation_models = pkgs.python312Packages.buildPythonPackage rec {
    pname = "segmentation-models-pytorch";
    version = "0.3.4";
    src = pkgs.fetchFromGitHub {
			owner = "qubvel-org";
			repo = "segmentation_models.pytorch";
			rev = "v0.3.4";
			sha256 = "M/7c/bItUe69dBiD47LFhhuD44648/R68iBXvAT0Jmc=";
		};
  };
  pretrainedmodels = pkgs.python312Packages.buildPythonPackage rec {
    pname = "pretrainedmodels";
    version = "0.7.4";
    src = pkgs.fetchFromGitHub {
      owner = "Cadene";
      repo = "pretrained-models.pytorch";
      rev = "8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0";
      sha256 = "OK865VBFRbsSZbEGHe1wLdkioj595YmLwaztwx2R6tE=";
    };
  };
  efficientnet = pkgs.python312Packages.buildPythonPackage rec {
    pname = "efficientnet-pytorch";
    version = "0.7.1";
    src = pkgs.fetchFromGitHub {
      owner = "lukemelas";
      repo = "EfficientNet-PyTorch";
      rev = "e047e4eb9e3ac1cb11e3efa69694c150293b16b1";
      sha256 = "RGOVhxjt0dFv3valneHjzZaF7m9JtC1MNkbh7MUGogo=";
    };
  };
in {
  # packages.${system}.example_derivation = pkgs.callPackage ./example_derivation.nix { };
  devShell.${system} = pkgs.mkShell {
    # packages = [
    #   self.packages.${system}.example_derivation 
    # ];
        packages = (with pkgs.python312Packages; [
      torchWithoutCuda 
      torchvision
      numpy
      opencv4
      # scikit-learn-extra
      scipy
      albumentations
    ]) ++ ([
      segmentation_models
      pretrainedmodels
      efficientnet
      pkgs.python312Packages.timm
    ]);

    shellHook = ''
      git_prompt() {
              local branch="$(git symbolic-ref HEAD 2> /dev/null | cut -d'/' -f3)"
              local master_or_main="$(git branch 2> /dev/null | awk -F ' +' '! /\(no branch\)/ {print $2}' | grep -E "(master|main)")"
              stats="$(git rev-list --count --left-right origin/$master_or_main... 2> /dev/null)"
              local down="$( echo $stats | awk '{ print $1}')"
              local up="$( echo $stats | awk '{ print $2}')"
              [ -n "$branch" ] && echo "($branch $down↓ $up↑)"
      }
      PS1='\[\033[01;32m\]\u@\h\[\033[00m\] \[\033[01;34m\]\w\[\033[00m\] \[\033[01;31m\](nix-develop)\[\033[00m\] \n $(git_prompt) '
    '';
  };
};
}
