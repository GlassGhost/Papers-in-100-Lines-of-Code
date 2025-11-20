# devShells.nix
{ pkgs ? import <nixpkgs> {}, appPythonEnv ? null }:

let
  # Import our wrapper 'rocm.nix' and force the gfx90c target
  customTorch = pkgs.python312Packages.callPackage ./rocm.nix {
    gpuTargets = [ "gfx90c" ];
  };

  # Define the default environment using the custom torch
  defaultPythonEnv = pkgs.python312.withPackages (ps: [ customTorch ]);

  # Use the passed environment if it exists, otherwise use our custom one
  finalPythonEnv = if appPythonEnv != null then appPythonEnv else defaultPythonEnv;

in pkgs.mkShell {
  packages = [
    finalPythonEnv
    
    # Python Bindings (torch removed to avoid conflicts)
    pkgs.python312Packages.pyside6
    pkgs.python312Packages.shiboken6
    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.numpy
    pkgs.python312Packages.opencv-python
    pkgs.python312Packages.stable-baselines3
    pkgs.python312Packages.tqdm
    pkgs.python312Packages.ale-py
    pkgs.python312Packages.gymnasium

    # ROCm & Vulkan
    pkgs.rocmPackages.rocm-runtime
    pkgs.rocmPackages.rocblas
    pkgs.rocmPackages.clr
    pkgs.rocmPackages.miopen
    pkgs.vulkan-loader
    pkgs.vulkan-headers
    pkgs.vulkan-tools

    # System Tools
    pkgs.ruff
    pkgs.uv
    pkgs.cmake
    pkgs.SDL2
    pkgs.wayland
    pkgs.libxkbcommon
    pkgs.pulseaudio
    pkgs.xorg.libXcomposite
    
    # Qt6
    pkgs.kdePackages.qtbase
    pkgs.kdePackages.qtdeclarative
    pkgs.kdePackages.qtsvg
  ];

  shellHook = ''
    if [[ $- == *i* ]]; then
      export PS1="[nix-dev-gfx90c:\u@\h:\w] "
    fi
    export LD_LIBRARY_PATH="${pkgs.rocmPackages.rocm-runtime}/lib:${pkgs.rocmPackages.rocblas}/lib:${pkgs.rocmPackages.clr}/lib:${pkgs.rocmPackages.miopen}/lib:$LD_LIBRARY_PATH"
    export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
  '';
}
# # devShells.nix
# # { pkgs, appPythonEnv }:
# # devShells.nix
# 
# # enter environment with
# # nix-shell devShells.nix
# 
# { pkgs ? import <nixpkgs> {},
#   appPythonEnv ? pkgs.python312.withPackages (ps: [ ps.torch ])
# }:
# 
# pkgs.mkShell {
#   packages = [
#     # Python bindings
#     pkgs.python312Packages.pyside6
#     pkgs.python312Packages.shiboken6
# #     pkgs.python312Packages.torch
#     pkgs.python312Packages.torchWithRocm
#     pkgs.python312Packages.matplotlib
#     pkgs.python312Packages.numpy
#     pkgs.python312Packages.opencv-python
#     pkgs.python312Packages.stable-baselines3
#     pkgs.python312Packages.tqdm
#     pkgs.python312Packages.ale-py
#     pkgs.python312Packages.gymnasium
# 
#     # ROCm for GPU support in PyTorch
#     pkgs.rocmPackages.rocm-runtime
#     pkgs.rocmPackages.rocblas
#     pkgs.rocmPackages.clr        # HIP runtime
#     pkgs.rocmPackages.miopen     # Optional ops, recommended
# 
#     # Vulkan
#     pkgs.vulkan-loader
#     pkgs.vulkan-headers
#     pkgs.vulkan-tools
# 
#     # Regular dev tools
#     appPythonEnv
#     pkgs.ruff
#     pkgs.uv
#     pkgs.cmake
#     pkgs.SDL2
#     pkgs.wayland
#     pkgs.libxkbcommon
#     pkgs.pulseaudio
#     pkgs.xorg.libXcomposite      # corrected path
# 
#     # Qt6 components for PySide6
#     pkgs.kdePackages.qtbase
#     pkgs.kdePackages.qtdeclarative
#     pkgs.kdePackages.qtsvg
#     pkgs.kdePackages.qttools
#     pkgs.kdePackages.qtmultimedia
#     pkgs.kdePackages.qtvirtualkeyboard
#     pkgs.kdePackages.qt3d
#   ];
# 
#   shellHook = ''
#     if [[ $- == *i* ]]; then
#       export ORIGINAL_PS1="$PS1"
#       export PS1="[nix-dev:\u@\h:\w] "
#     fi
# 
#     # Add ROCm to library path for runtime
#     export LD_LIBRARY_PATH="${pkgs.rocmPackages.rocm-runtime}/lib:${pkgs.rocmPackages.rocblas}/lib:${pkgs.rocmPackages.clr}/lib:${pkgs.rocmPackages.miopen}/lib:$LD_LIBRARY_PATH"
# 
#     # Vulkan layer if needed
#     export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
# 
#     # If using zsh, you would do something similar for PROMPT:
#     # export PROMPT="[flake-shell] $PROMPT"
#   '';
# }
