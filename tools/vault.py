"""Encrypted store for dynamic credentials (spotify refresh token, etc.).

Static config like API keys lives in .env; anything the app *acquires* at
runtime goes here, encrypted with a Fernet key from VAULT_KEY. Everything is
secret-granular: each value is independently encrypted, and the whole file is
rewritten atomically (temp file + os.replace) so a crash mid-write can't
corrupt the store.

Deliberately fails fast: constructing a Vault without VAULT_KEY raises right
away, so the failure surfaces at startup instead of a confusing decode error
on first use.
"""

import json
import logging
import os
import tempfile

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

VAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_data")
VAULT_FILE = os.path.join(VAULT_DATA_DIR, "secrets.enc.json")


class Vault:
    """Fernet-encrypted secrets backed by a single JSON file."""

    def __init__(self, key: str | None = None, file_path: str = VAULT_FILE):
        if key is None:
            key = os.getenv("VAULT_KEY")
        if not key:
            raise RuntimeError(
                "VAULT_KEY is not set. Copy .env.example to .env and set a "
                "Fernet key there (generate one with: "
                'python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())").'
            )
        self._fernet = Fernet(key.encode())
        self.file_path = file_path
        directory = os.path.dirname(file_path) or "."
        os.makedirs(directory, exist_ok=True)

    def _load(self) -> dict:
        if not os.path.isfile(self.file_path):
            return {}
        with open(self.file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict) -> None:
        directory = os.path.dirname(self.file_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            os.replace(tmp_path, self.file_path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def set(self, key: str, value: str) -> None:
        data = self._load()
        data[key] = self._fernet.encrypt(value.encode()).decode()
        self._save(data)

    def get(self, key: str) -> str | None:
        stored = self._load().get(key)
        if stored is None:
            return None
        try:
            return self._fernet.decrypt(stored.encode()).decode()
        except InvalidToken:
            logger.error(
                "Vault secret %r failed to decrypt (VAULT_KEY changed?). Returning None.",
                key,
            )
            return None

    def delete(self, key: str) -> None:
        data = self._load()
        if key in data:
            del data[key]
            self._save(data)