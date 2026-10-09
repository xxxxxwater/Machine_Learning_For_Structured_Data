# GitHub Packages: MLSD container image

MLSD is a Python library, but **GitHub Packages is not a PyPI registry**. The
GitHub Packages artifact for this project is an OCI/Docker image published to
GitHub Container Registry (GHCR), containing the **same Python source as the
existing `v0.2.0` Release tag**.

- Registry image: `ghcr.io/xxxxxwater/mlsd-structured-data`
- Tags: `v0.2.0` (fixed release), `latest` (current package tag)
- Source: https://github.com/xxxxxwater/Machine_Learning_For_Structured_Data
- Package listing: https://github.com/xxxxxwater/Machine_Learning_For_Structured_Data/packages
- Publication workflow: [publish-ghcr.yml](../.github/workflows/publish-ghcr.yml)
- Conventional Python packages (wheel and sdist): [v0.2.0 Release](https://github.com/xxxxxwater/Machine_Learning_For_Structured_Data/releases/tag/v0.2.0)

## Pull and run

If the package is public:

```bash
docker pull ghcr.io/xxxxxwater/mlsd-structured-data:v0.2.0
docker run --rm ghcr.io/xxxxxwater/mlsd-structured-data:v0.2.0
```

By default, this container runs a synthetic mixed-data ML training demonstration.
To run your own script, mount a local working directory and invoke Python explicitly:

```bash
docker run --rm -v "$PWD:/work:ro" \
  ghcr.io/xxxxxwater/mlsd-structured-data:v0.2.0 \
  python /work/my_script.py
```

The image runs as a non-root user. Scripts in a bind mount need to be readable
by that user. No API keys, SSH credentials or datasets are included.

## Private package authentication

GitHub initially makes new GHCR packages private. If the image is private,
authenticate using a **classic personal access token** with `read:packages`
permission. Never commit a token or paste it into logs.

```bash
echo "$CR_PAT" | docker login ghcr.io -u xxxxxwater --password-stdin
docker pull ghcr.io/xxxxxwater/mlsd-structured-data:v0.2.0
```

For public access, a package administrator can use **Package settings →
Change visibility → Public** on the package page. Package visibility is
separate from repository visibility.

## Publishing and verification

`publish-ghcr.yml` checks out the pinned `v0.2.0` release, executes the
Python test suite and example, builds an image from the release source,
runs the container's example, authenticates via the scoped GitHub Actions
`GITHUB_TOKEN` and pushes both GHCR tags. It verifies both remote manifests.

The image includes OCI source, description, version and MIT license labels
for proper linkage to the GitHub repository. The workflow may be manually
re-run from GitHub Actions and is only automatically run when its own
workflow YAML is updated.

This is a runtime/container distribution, **not** an additional upload of
`.whl` files to GitHub Packages, since no PyPI-compatible GitHub Packages
registry currently exists.
