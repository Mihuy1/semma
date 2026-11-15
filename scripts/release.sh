#!/bin/bash
# Releases the version in pyproject.toml: tags the current commit v<version> and pushes the branch and
# the tag. The tag starts the Release workflow on GitHub (.github/workflows/release.yml), which builds
# the .rpm and the .deb, checks each in a clean container and publishes them as the release, with the
# commits since the last tag as its notes. Follow it with `gh run watch`.
# Refuses when the working tree has changes or that version is already released.
# Needs gh (logged in).
#   scripts/release.sh
# To build and check the packages on this machine instead: scripts/package.sh, then
# scripts/check-package.sh dist/<package>.
set -euo pipefail
cd "$(dirname "$0")/.."

version=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
tag=v$version
if [[ -n "$(git status --porcelain)" ]]; then
  echo "the working tree has changes: commit them first" >&2
  exit 1
fi
if gh release view "$tag" > /dev/null 2>&1; then
  echo "$tag is already released: raise the version in pyproject.toml" >&2
  exit 1
fi
if git ls-remote --exit-code --tags origin "refs/tags/$tag" > /dev/null; then
  echo "$tag is already on GitHub but not released: re-run its failed Release workflow instead" >&2
  exit 1
fi

git tag -f "$tag" > /dev/null   # -f: a tag left here by a push that failed
git push --quiet origin HEAD "$tag"
echo "pushed $tag: the Release workflow builds and publishes it (gh run watch)"
