"""Access a dataset given as a Hugging Face link, an ``owner/name`` id, or a local folder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HF_LINK = re.compile(
    r"^(?:https?://(?:www\.)?huggingface\.co/)?(?:datasets/)?([\w.-]+/[\w.-]+)"
    r"(?:/(?:tree|blob|resolve)/([^/]+)(?:/(.*))?)?/?$"
)


class DatasetLinkError(ValueError):
    pass


@dataclass
class Repo:
    spec: str
    download_dir: Path
    repo_id: str | None = None
    revision: str | None = None
    subdir: str = ""
    local: Path | None = None

    def __post_init__(self) -> None:
        self._files: list[str] | None = None
        p = Path(self.spec).expanduser()
        if p.exists():
            self.local = p.resolve()
            return
        m = _HF_LINK.match(self.spec.strip())
        if not m:
            raise DatasetLinkError(f"Not a Hugging Face dataset link or an existing folder: {self.spec}")
        self.repo_id, self.revision = m.group(1), m.group(2)
        self.subdir = (m.group(3) or "").strip("/")

    @property
    def name(self) -> str:
        return self.repo_id or str(self.local)

    @property
    def root(self) -> Path:
        if self.local:
            return self.local
        return (self.download_dir / self.repo_id.replace("/", "__")).resolve()

    def files(self) -> list[str]:
        """Repo-relative file paths (listing only, nothing is downloaded)."""
        if self._files is None:
            if self.local:
                files = sorted(p.relative_to(self.local).as_posix() for p in self.local.rglob("*") if p.is_file())
            else:
                from huggingface_hub import list_repo_files

                files = sorted(list_repo_files(self.repo_id, repo_type="dataset", revision=self.revision))
            if self.subdir:
                files = [f for f in files if f == self.subdir or f.startswith(self.subdir + "/")]
            self._files = files
        return self._files

    def fetch(self, filenames: list[str]) -> None:
        """Download specific files (skips ones already present)."""
        if self.local:
            return
        from huggingface_hub import hf_hub_download

        for f in filenames:
            if not (self.root / f).exists():
                hf_hub_download(self.repo_id, f, repo_type="dataset", revision=self.revision, local_dir=self.root)

    def fetch_patterns(self, patterns: list[str]) -> None:
        """Download every file matching the glob patterns (resumable)."""
        if self.local:
            return
        from huggingface_hub import snapshot_download

        snapshot_download(self.repo_id, repo_type="dataset", revision=self.revision,
                          local_dir=self.root, allow_patterns=patterns)
