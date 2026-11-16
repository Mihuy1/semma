"""The Screenshots section's settings (ocr.enabled, ocr.languages) as plain values. No Qt here."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ... import ocr
from .context import Changes, Problem, SettingsContext

SECTION = "ocr"
DEFAULT_ENABLED = False
MAX_LANGUAGES = ocr.MAX_LANGUAGES
EXAMPLE_LANGUAGE = "deu"


@dataclass(frozen=True)
class OcrFacts:
    """Snapshot of the system's Tesseract installation, injectable for tests."""

    library_available: bool
    installed_languages: tuple[str, ...]
    default_languages: tuple[str, ...]
    install_hint: Callable[[str | None], str] = ocr.install_hint
    language_name: Callable[[str], str] = ocr.language_name

    @classmethod
    def detect(cls) -> OcrFacts:
        installed = tuple(ocr.installed_languages())
        return cls(
            library_available=ocr.library_available(),
            installed_languages=installed,
            default_languages=tuple(ocr.default_languages(installed)),
            install_hint=ocr.install_hint,
            language_name=ocr.language_name,
        )


def normalise_languages(codes: Iterable[str | None]) -> list[str]:
    """Deduplicate while keeping order, drop blanks/None, cap at MAX_LANGUAGES."""
    out: list[str] = []
    for raw in codes:
        if not isinstance(raw, str):
            continue
        code = raw.strip().lower()
        if code and code not in out:
            out.append(code)
            if len(out) >= MAX_LANGUAGES:
                break
    return out


def language_choices(configured: Sequence[str], facts: OcrFacts) -> list[tuple[str, str | None]]:
    """Combo-box (label, code) pairs: None first, then installed and configured languages sorted by name."""
    installed = set(facts.installed_languages)
    codes = set(installed) | set(normalise_languages(configured))
    items: list[tuple[str, str]] = []
    for code in codes:
        name = facts.language_name(code)
        label = name if code in installed else f"{name} (not installed)"
        items.append((label, code))
    items.sort(key=lambda pair: (facts.language_name(pair[1]).casefold(), pair[1]))
    return [("None", None), *items]


@dataclass
class ScreenshotsValues:
    enabled: bool = DEFAULT_ENABLED
    languages: list[str] = field(default_factory=list)
    default_languages: list[str] | None = field(default=None, compare=False, repr=False)

    def copy(self) -> ScreenshotsValues:
        return ScreenshotsValues(
            enabled=bool(self.enabled),
            languages=list(self.languages),
            default_languages=(
                list(self.default_languages) if self.default_languages is not None else None
            ),
        )


DEFAULTS = ScreenshotsValues()


def load(ctx: SettingsContext, facts: OcrFacts | None = None) -> ScreenshotsValues:
    facts = facts if facts is not None else OcrFacts.detect()
    cfg = ctx.config
    auto = normalise_languages(facts.default_languages)
    langs = list(auto) if cfg.ocr_languages is None else normalise_languages(cfg.ocr_languages)
    return ScreenshotsValues(
        enabled=bool(cfg.ocr_enabled),
        languages=langs,
        default_languages=auto,
    )


def _auto_languages(
    saved: ScreenshotsValues,
    edited: ScreenshotsValues,
    facts: OcrFacts | None,
) -> list[str]:
    if facts is not None:
        return normalise_languages(facts.default_languages)
    if edited.default_languages is not None:
        return normalise_languages(edited.default_languages)
    if saved.default_languages is not None:
        return normalise_languages(saved.default_languages)
    return normalise_languages(OcrFacts.detect().default_languages)


def changes(
    saved: ScreenshotsValues,
    edited: ScreenshotsValues,
    facts: OcrFacts | None = None,
) -> Changes:
    """Only keys the user changed; returning to the default yields None so the key is removed."""
    ocr_out: dict[str, object] = {}

    old_enabled = bool(saved.enabled)
    new_enabled = bool(edited.enabled)
    if new_enabled != old_enabled:
        ocr_out["enabled"] = None if new_enabled is DEFAULT_ENABLED else new_enabled

    old_langs = normalise_languages(saved.languages)
    new_langs = normalise_languages(edited.languages)
    if new_langs != old_langs:
        auto = _auto_languages(saved, edited, facts)
        ocr_out["languages"] = None if (not new_langs or new_langs == auto) else new_langs

    return {SECTION: ocr_out} if ocr_out else {}


def problems(
    values: ScreenshotsValues,
    ctx: SettingsContext | None = None,
    facts: OcrFacts | None = None,
) -> list[Problem]:
    facts = facts if facts is not None else OcrFacts.detect()
    installed = set(facts.installed_languages)
    langs = normalise_languages(values.languages)
    out: list[Problem] = []

    for code in langs:
        if code not in installed:
            name = facts.language_name(code)
            hint = facts.install_hint(code)
            if hint:
                text = f"The {name} language pack is not installed. Install it with: {hint}"
            else:
                text = f"The {name} language pack is not installed."
            out.append(Problem(text, blocking=False, code=f"missing-language:{code}"))

    if values.enabled:
        usable = [code for code in langs if code in installed]
        if not facts.library_available:
            out.append(
                Problem(
                    "Tesseract is not installed, so no text will be read from screenshots.",
                    blocking=False,
                    code="no-usable-language",
                )
            )
        elif not usable:
            if not langs:
                msg = "No language is chosen, so no text will be read from screenshots."
            else:
                msg = "None of the chosen languages are installed, so no text will be read from screenshots."
            out.append(Problem(msg, blocking=False, code="no-usable-language"))

    return out
