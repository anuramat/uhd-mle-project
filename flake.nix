{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-unstable";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in rec {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages (python-pkgs:
              with python-pkgs; [
                jupyter
                jupytext
                numpy
                scipy
                scikit-learn
                tqdm
                pygame
                # # this conflicts with lightning
                # torchWithCuda
                # # this "fix" doesn't work for some reason
                # pytorch-lightning.override
                # {torch = torchWithCuda;}
                pytorch-lightning # import as `pytorch_lightning`
                pytest
              ]))
          ];
        };
      }
    );
}
