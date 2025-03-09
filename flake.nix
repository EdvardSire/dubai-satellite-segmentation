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
    config.allowUnfree = true;
  };
  # simElevatorServer = import ./simElevatorServer.nix;
in {
  devShell.${system} = pkgs.mkShell {
    # buildInputs = [
    #   simElevatorServer
    # ];
    packages = (with pkgs.python312Packages; [
      torchWithoutCuda 
      torchvision
      numpy
      opencv4
      # scikit-learn-extra
      scipy
      albumentations
    ]) ++ (with pkgs; [
      gitMinimal
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
