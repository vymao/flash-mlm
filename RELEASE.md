# Release workflow

This repository supports both:

- install from GitHub tags
- install from PyPI (`pip install flash-mlm`)

PyPI publishing is automated by GitHub Actions on tag pushes matching `v*`
(`.github/workflows/python-package.yml`).

The workflow gates publishing on a Linux unit-test pass
(`pytest -q src/flash_mlm/test_host_utils.py`).

On tag-triggered runs, the workflow automatically sets `project.version` from the
tag (for example, `v1.0.4` -> `1.0.4`) before building.

## 0) One-time PyPI setup (required)

Use PyPI Trusted Publishing for this repo:

1. Create the `flash-mlm` project on PyPI (or claim it if already created).
2. In PyPI project settings, add a Trusted Publisher with:
   - Owner: `vymao`
   - Repository: `flash-mlm`
   - Workflow filename: `python-package.yml`
   - Environment: `pypi`
3. In GitHub repo settings, create environment `pypi`.
   - Optional but recommended: add required reviewers.

## 1) Commit and push

```bash
git add -A
git commit -m "Release vX.Y.Z"
git push
```

## 2) Create and push tag

```bash
git tag -a vX.Y.Z -m "flash-mlm vX.Y.Z"
git push origin vX.Y.Z
```

This triggers the `Publish to PyPI` workflow.

## 3) Verify package on PyPI

Check workflow run:

- `https://github.com/vymao/flash-mlm/actions`

Then verify install:

```bash
pip install --upgrade flash-mlm==X.Y.Z
python -c "import flash_mlm; print('flash_mlm import ok')"
```

## 4) Verify install from Git tag (optional)

```bash
pip install "flash-mlm @ git+https://github.com/vymao/flash-mlm.git@vX.Y.Z"
```

## 5) (Optional) GitHub Release page

Create a GitHub Release for `vX.Y.Z` with changelog notes and example install command.
