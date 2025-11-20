
## 🖥️ (Build Host)**binary cache server** Setup

**Server = build + serve binaries with nix-serve. Clients = configure substituters + trust public key.** This way, clients skip compilation and fetch prebuilt packages directly.

**Server builds packages** (either manually with `nix-build` or automatically via CI/CD). Hosts prebuilt store paths of built packages over HTTP/HTTPS with signing keys, while clients add the cache URL and public key to their Nix configuration.

If not found, the client builds locally (or you can configure remote builders via SSH for distributed builds).

1. **Generate signing keys**  
   ```bash
   nix-store --generate-binary-cache-key cache-name-1 ./secret-key ./public-key
   ```
   - `secret-key` stays on the server.  
   - `public-key` is shared with clients.

2. **Enable Nix binary cache service**  
   On NixOS, add to `/etc/nixos/configuration.nix`:
   ```nix
   services.nix-serve.enable = true;
   services.nix-serve.secretKeyFile = "/etc/nix/secret-key";
   ```
   This runs `nix-serve` which exposes `/nix/store` objects over HTTP.

3. **Open the port**  
   By default, `nix-serve` listens on port 5000. You can reverse-proxy it with Nginx/Apache to serve over HTTPS if desired.

## 💻 Client Setup

**Clients request packages** → Nix checks the cache → downloads prebuilt binaries if available.

1. **Add cache URL and public key**  
   In `/etc/nix/nix.conf` (or `~/.config/nix/nix.conf`):
   ```ini
   substituters = https://cache.nixos.org http://your-server:5000
   trusted-public-keys = cache-name-1:PUBLIC_KEY_STRING cache.nixos.org-1:... 
   ```
   Replace `PUBLIC_KEY_STRING` with the contents of your generated `public-key`.

2. **Test fetching from cache**  
   On the client, try installing a package built on the server:
   ```bash
   nix-build '<nixpkgs>' -A hello
   ```
   If the server has already built it, the client will fetch the binary instead of compiling.

## Tips
- Use **remote builders** (`nix.buildMachines`) if you want clients to offload builds to the server instead of just fetching binaries.  
- For larger setups, consider **S3/MinIO-based caches** or **Cachix**, which simplify sharing binaries across teams.  
- Always keep signing keys secure; only distribute the public key.

