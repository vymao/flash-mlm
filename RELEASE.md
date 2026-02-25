# Release workflow

This repository is installable directly from GitHub tags.

## 1) Update version

Edit `pyproject.toml`:

- `project.version = "X.Y.Z"`

## 2) Commit and push

```bash
git add -A
git commit -m "Release vX.Y.Z"
git push
```

## 3) Create and push tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

## 4) Verify install from tag

```bash
pip install "flash-mlm @ git+https://github.com/vymao/flash-mlm.git@vX.Y.Z"
```

## 5) (Optional) GitHub Release page

Create a GitHub Release for `vX.Y.Z` with changelog notes and example install command.
