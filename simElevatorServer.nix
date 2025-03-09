{ stdenv, pkgs }:

stdenv.mkDerivation {
  pname = "sim-elevator-server";
  version = "1.0.0";

  src = pkgs.fetchgit {
    url = "https://github.com/TTK4145/Simulator-v2.git";
    rev = "86a0a3ff53a77aa7517ebcbd0940a75d7a1ebc11";
    sha256 = "sha256-gtVyfFLp4UO9oNaCP5a7upfqvF4GRCnQHpVA8gx7fZg=";
  };

  nativeBuildInputs = [ pkgs.makeWrapper ];
  buildInputs = [ pkgs.ldc ];

  buildPhase = ''
    ldc2 -w -g src/sim_server.d src/timer_event.d -ofsimElevatorServer
  '';

  installPhase = ''
    mkdir -p $out/bin
    cp simElevatorServer $out/bin/
    cp simulator.con $out/bin/
  '';
}

