
# Deep Q Nets

We need to get this running properly.

#### Build PyTorch from source with gfx90c support

We you want GPU acceleration:
- We could Clone PyTorch and build it with ROCm, explicitly enabling `gfx90c` in the build flags:
  ```bash
  USE_ROCM=1 HCC_AMDGPU_TARGET=gfx90c python setup.py install
  ```
- This compiles kernels for your GPU. It’s more work, but it’s the only way to get HIP working on unsupported consumer GPUs.

#### bute we are using nix

Based on the `test_output.log`, the error `RuntimeError: HIP error: no kernel image is available for execution on the device` indicates that the PyTorch binary was compiled successfully for *some* AMD GPUs, but not for your specific architecture (`gfx90c`, which is common in Ryzen APUs and the Steam Deck).

Here are the modifications needed for `rocm.nix` to force the build to include your specific GPU kernel, along with a minor fix for the Python script.

### 1\. Modify `rocm.nix`

You need to explicitly define the `gpuTargets`. The cleanest way to do this within the file (without relying on external overrides) is to change the default value of the `gpuTargets` argument at the very top of the file.

This change will ensure `PYTORCH_ROCM_ARCH` is set to `gfx90c` during the build process, forcing the compiler to generate the correct kernels for your hardware.

**File:** `rocm.nix`
**Location:** Line 103

**Change this:**

```nix
  rocmPackages,
  gpuTargets ? [ ],  # <--- Original Line 103

  vulkanSupport ? false,
```

**To this:**

```nix
  rocmPackages,
  gpuTargets ? [ "gfx90c" ], # <--- Modified: Force gfx90c target

  vulkanSupport ? false,
```

**Why this works:**
By populating this list, the logic at (`if gpuTargets != [ ] then...`) takes precedence over the complex detection logic further down. It sets the `gpuTargetString` to `gfx90c`, which is then exported as `PYTORCH_ROCM_ARCH` at **Line 333**.

-----


### Summary of Next Steps

```
[owner@nixos:~/Papers-in-100-Lines-of-Code/Playing_Atari_with_Deep_Reinforcement_Learning]$ ls
breakout.bin     concat.md      dqn.gif  log_macro.sh     requirements.txt  test_output.log
commands.sh      concat.py      dqn.py   README.md        rocm.nix          test.sh
concat_macro.sh  devShells.nix  Imgs     requirements.md  roms

```

We need to modify our devShells.nix file to override our pytorchwithrocm import to use our fixed default.nix for pytorchwithrocm `rocm.nix` file instead and build our package.

Remember this is a test for system so we can use it to get more tokens per second, so we can have more tokens.

