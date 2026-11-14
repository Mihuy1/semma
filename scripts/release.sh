#!/bin/bash
# Publishes the version in pyproject.toml as a GitHub release with the .rpm and the .deb attached:
# builds both (scripts/package.sh), checks each in a clean container (scripts/check-package.sh), then
# tags the current commit v<version>, pushes the branch and the tag, and creates the release.
# Refuses when the working tree has changes or that version is already released.
# Needs gh (logged in), and what scripts/package.sh needs.
#   scripts/release.sh            the release notes are the commits since the last tag
#   scripts/release.sh --draft    as a draft, to look at before it is published
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

scripts/package.sh
rpm=dist/semma-$version-1.x86_64.rpm
deb=dist/semma_$version-1_amd64.deb
scripts/check-package.sh "$rpm"
scripts/check-package.sh "$deb"

git tag -f "$tag" > /dev/null   # -f: a tag left by a run that failed after this point
git push --quiet origin HEAD "$tag"
# A version with a suffix (0.2.0-beta.1) is a prerelease.
flags=()
[[ $version == *-* ]] && flags+=(--prerelease)
[[ ${1:-} == --draft ]] && flags+=(--draft)
gh release create "$tag" "$rpm" "$deb" --title "Semma $version" --generate-notes --verify-tag "${flags[@]}"
