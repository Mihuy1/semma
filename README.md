# Semma

Semantic file search for KDE KRunner and GNOME Shell search, powered by EmbeddingGemma 2. A background service indexes your files locally, and you search them by meaning from the desktop's search.

## Install

Packages for x86-64 Fedora 41+, Ubuntu 24.04+ and Debian 13+:

```sh
sudo dnf install ./semma-0.3.0-1.x86_64.rpm     # Fedora
sudo apt install ./semma_0.3.0-1_amd64.deb      # Ubuntu, Debian
```

Then open **Semma** from the application menu and go through the setup wizard. Semma does not index anything until you finish the wizard. The model (865 MB) downloads in the background the first time.

## Text in screenshots

Semma can also read the text in your screenshots with Tesseract, so you can find a screenshot by the words in it. This is off by default; turn it on in Semma's settings (or with `enabled = true` under `[ocr]` in `~/.config/semma/config.toml`). It reads your system's language and English when their language data is installed. To add a language, install its Tesseract language pack, for example `sudo dnf install tesseract-langpack-deu` (Fedora) or `sudo apt install tesseract-ocr-deu` (Ubuntu, Debian).

## Uninstall

```sh
sudo dnf remove semma     # Fedora
sudo apt remove semma     # Ubuntu, Debian
```

Your index and the model stay in `~/.local/share/semma`. Delete that folder to remove them too.

## From source

Needs [uv](https://docs.astral.sh/uv/).

```sh
uv run semma install      # add Semma to KRunner or GNOME search and start the service
uv run semma uninstall
```

`install` detects KDE or GNOME; pass `--desktop kde|gnome` to choose. On GNOME, log out and back in afterwards for the search provider to appear.

- Search from the terminal: `uv run semma search <query>`
- Open the window: `uv run semma gui`
- Run the tests: `uv run pytest -q`
- Build the packages: `scripts/package.sh` (needs `curl`, and `podman` or `docker`); they land in `dist/`

## License

MIT, see [LICENSE](LICENSE).
