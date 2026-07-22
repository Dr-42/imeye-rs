{
  description = "imeye-rs - GPU image viewer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        
        # Extracted so we can feed it to both the shell and the package wrapper
        runtimeLibs = with pkgs; [
          libGL
          mesa
          wayland
          libxkbcommon
          libX11
          libXcursor
          libXi
          libXrandr
          udev
        ];
      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "imeye-rs";
          version = "0.1.0";

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          # makeWrapper added to nativeBuildInputs
          nativeBuildInputs = [ pkgs.pkg-config pkgs.makeWrapper ];

          buildInputs = runtimeLibs;

          # Wrap the final binary with the dynamic libraries so `dlopen` finds them
          postInstall = ''
            wrapProgram $out/bin/imeye-rs \
              --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath runtimeLibs}"
          '';
        };

        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            rustc
            cargo
            pkg-config
          ] ++ runtimeLibs;

          # The missing hook that fixes `cargo run` inside `nix develop`
          shellHook = ''
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath runtimeLibs}
            export RUSTFLAGS="-C link-arg=-Wl,-rpath,${pkgs.lib.makeLibraryPath runtimeLibs}"
          '';
        };
      });
}
