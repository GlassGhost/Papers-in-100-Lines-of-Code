# rocm.nix
# This wrapper takes the standard torchWithRocm package and 
# re-configures it to build specifically for gfx90c.

{ torchWithRocm, gpuTargets ? [ "gfx90c" ], ... }:

torchWithRocm.override {
  inherit gpuTargets;
}