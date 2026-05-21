"""Honcho-backed memory service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_HONCHO_BASE_URL = "http://localhost:8000"
DEFAULT_WORKSPACE_ID = "huesae-agents"
DEFAULT_USER_PEER_ID = "local-user"
DEFAULT_ASSISTANT_PEER_ID = "huesae-main-agent"
DEFAULT_STATE_PATH = Path.home() / ".huesaeagents" / "honcho" / "session.json"
DEFAULT_WORKSPACE_STATE_PATH = Path(__file__).resolve().parents[4] / ".state" / "honcho" / "session.json"
DEFAULT_CONTEXT_TOKENS = 1600
DEFAULT_SEARCH_TOP_K = 8
DEFAULT_MAX_CONCLUSIONS = 24
NO_MEMORY_TEXT = "暂无可用用户记忆。"
MEMORY_UNAVAILABLE_TEXT = "记忆服务暂时不可用，本轮不要依赖历史记忆。"


class HonchoMemoryService:
    """Thin Honcho adapter for persistent single-user terminal memory."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
        user_peer_id: str | None = None,
        assistant_peer_id: str | None = None,
        state_path: str | Path | None = None,
        client: Any | None = None,
    ):
        self.base_url = (base_url or os.getenv("HONCHO_BASE_URL") or DEFAULT_HONCHO_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("HONCHO_API_KEY")
        self.workspace_id = workspace_id or os.getenv("HONCHO_WORKSPACE_ID") or DEFAULT_WORKSPACE_ID
        self.user_peer_id = user_peer_id or os.getenv("HONCHO_USER_PEER_ID") or DEFAULT_USER_PEER_ID
        self.assistant_peer_id = (
            assistant_peer_id
            or os.getenv("HONCHO_ASSISTANT_PEER_ID")
            or DEFAULT_ASSISTANT_PEER_ID
        )
        self.state_path = Path(state_path or os.getenv("HONCHO_STATE_PATH") or DEFAULT_STATE_PATH)
        self.client = client
        self.session = None
        self.user_peer = None
        self.assistant_peer = None
        self.enabled = False
        self.status = "not initialized"

    def initialize(self) -> bool:
        """Connect to Honcho and restore or create the persistent session."""
        try:
            self._check_server()
            if self.client is None:
                self.client = self._create_client()
            self.user_peer = self.client.peer(self.user_peer_id)
            self.assistant_peer = self.client.peer(self.assistant_peer_id)
            self._configure_peer_observation()
            self.session = self._get_or_create_session()
            if hasattr(self.session, "add_peers"):
                self.session.add_peers(self._session_peers_with_config())
            self.enabled = True
            self.status = f"enabled: {self.session_id}"
            return True
        except Exception as exc:
            self.enabled = False
            self.status = f"disabled: {exc}"
            return False

    @property
    def session_id(self) -> str | None:
        """Return the current Honcho session id, if available."""
        return self._session_id

    def get_context(self, user_input: str | None = None, max_tokens: int | None = None) -> str:
        """Return Honcho's optimized short and long-term memory context."""
        if not self.enabled or self.session is None:
            return NO_MEMORY_TEXT

        try:
            query = str(user_input or "").strip() or None
            context = self.session.context(
                summary=True,
                tokens=max_tokens or _env_int("HONCHO_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS),
                peer_target=self.user_peer_id,
                search_query=query,
                search_top_k=_env_int("HONCHO_SEARCH_TOP_K", DEFAULT_SEARCH_TOP_K),
                max_conclusions=_env_int("HONCHO_MAX_CONCLUSIONS", DEFAULT_MAX_CONCLUSIONS),
                include_most_frequent=True,
            )
            text = self._context_to_text(context)
            return text or NO_MEMORY_TEXT
        except Exception as exc:
            self.status = f"context failed: {exc}"
            return MEMORY_UNAVAILABLE_TEXT

    def store_exchange(self, user_input: str, assistant_response: str) -> bool:
        """Persist a completed user/assistant exchange into Honcho."""
        if not self.enabled or self.session is None:
            return False

        messages = []
        if user_input.strip():
            messages.append(self._peer_message(self.user_peer, user_input))
        if assistant_response.strip():
            messages.append(self._peer_message(self.assistant_peer, assistant_response))
        messages = [message for message in messages if message is not None]
        if not messages:
            return False

        try:
            self.session.add_messages(messages)
            return True
        except Exception as exc:
            self.status = f"store failed: {exc}"
            return False

    def _create_client(self):
        """Create the official Honcho SDK client lazily."""
        try:
            from honcho import Honcho
        except ImportError as exc:
            raise RuntimeError("missing optional dependency honcho-ai") from exc

        kwargs = {"workspace_id": self.workspace_id}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            return Honcho(**kwargs)
        except TypeError:
            kwargs.pop("base_url", None)
            try:
                return Honcho(**kwargs)
            except TypeError:
                kwargs.pop("api_key", None)
                return Honcho(**kwargs)

    def _check_server(self) -> None:
        """Perform a small startup check before creating SDK objects."""
        url = f"{self.base_url}/health"
        try:
            with urlopen(Request(url), timeout=2) as response:
                if response.status >= 500:
                    raise RuntimeError(f"Honcho health check failed: HTTP {response.status}")
        except URLError as exc:
            raise RuntimeError(f"Honcho is not reachable at {url}") from exc

    def _configure_peer_observation(self) -> None:
        """Best-effort peer-level reasoning configuration for long-term memory."""
        try:
            from honcho.api_types import PeerConfig
        except ImportError:
            return

        try:
            if hasattr(self.user_peer, "set_configuration"):
                self.user_peer.set_configuration(PeerConfig(observe_me=True))
            if hasattr(self.assistant_peer, "set_configuration"):
                self.assistant_peer.set_configuration(PeerConfig(observe_me=False))
        except Exception as exc:
            self.status = f"peer configuration skipped: {exc}"

    def _session_peers_with_config(self):
        """Return session peers with observation flags when the SDK supports them."""
        try:
            from honcho.session import SessionPeerConfig
        except ImportError:
            return [self.user_peer, self.assistant_peer]

        user_config = SessionPeerConfig(observe_me=True, observe_others=False)
        assistant_config = SessionPeerConfig(observe_me=False, observe_others=True)
        return [
            (self.user_peer, user_config),
            (self.assistant_peer, assistant_config),
        ]

    def _get_or_create_session(self):
        """Restore the persistent session id or create and remember a new one."""
        self._session_id = self._read_session_id() or self._default_session_id()
        return self._session_from_id(self._session_id)

    def _session_from_id(self, session_id: str | None):
        if not session_id:
            return None
        try:
            session = self.client.session(session_id)
        except TypeError:
            session = self.client.session(id=session_id)
        self._write_session_id(session_id)
        return session

    def _read_session_id(self) -> str | None:
        for path in self._state_path_candidates():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue

            if data.get("workspace_id") != self.workspace_id:
                continue
            if data.get("user_peer_id") != self.user_peer_id:
                continue
            session_id = data.get("session_id")
            session_id = str(session_id).strip() if session_id is not None else ""
            if session_id:
                self.state_path = path
                return session_id
        return None

    def _write_session_id(self, session_id: str | None) -> None:
        if not session_id:
            return
        data = {
            "workspace_id": self.workspace_id,
            "user_peer_id": self.user_peer_id,
            "assistant_peer_id": self.assistant_peer_id,
            "session_id": session_id,
        }
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        for path in self._state_path_candidates():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload, encoding="utf-8")
                self.state_path = path
                return
            except OSError:
                continue

    def _default_session_id(self) -> str:
        """Build a stable session id for the single-user terminal flow."""
        raw = f"{self.workspace_id}:{self.user_peer_id}"
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)

    def _state_path_candidates(self) -> list[Path]:
        """Return writable state file candidates in priority order."""
        candidates = [self.state_path, DEFAULT_WORKSPACE_STATE_PATH]
        deduped: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped

    @staticmethod
    def _peer_message(peer: Any, content: str):
        if peer is None:
            return None
        if hasattr(peer, "message"):
            return peer.message(content)
        return {"peer_id": getattr(peer, "id", None), "content": content}

    def _context_to_text(self, context: Any) -> str:
        if context is None:
            return ""
        if isinstance(context, str):
            return context.strip()
        session_text = self._session_context_to_text(context)
        if session_text:
            return session_text
        if hasattr(context, "to_text"):
            return str(context.to_text()).strip()
        if hasattr(context, "to_openai"):
            assistant = self.assistant_peer or self.assistant_peer_id
            try:
                openai_messages = context.to_openai(assistant=assistant)
            except TypeError:
                openai_messages = context.to_openai()
            text = _messages_to_text(openai_messages)
            if text:
                return text
        if hasattr(context, "messages"):
            return _messages_to_text(getattr(context, "messages"))
        if hasattr(context, "__str__"):
            return str(context).strip()
        return ""

    def _session_context_to_text(self, context: Any) -> str:
        """Format Honcho SessionContext into a compact prompt block."""
        sections: list[str] = []

        peer_representation = _clean_text(getattr(context, "peer_representation", None))
        if peer_representation:
            sections.append(f"长期记忆（用户画像与结论）:\n{peer_representation}")

        peer_card = getattr(context, "peer_card", None)
        peer_card_text = _peer_card_to_text(peer_card)
        if peer_card_text:
            sections.append(f"用户卡片:\n{peer_card_text}")

        summary = getattr(context, "summary", None)
        summary_text = _clean_text(getattr(summary, "content", None))
        if summary_text:
            sections.append(f"会话摘要:\n{summary_text}")

        messages_text = _messages_to_text(getattr(context, "messages", None))
        if messages_text:
            sections.append(f"相关近期消息:\n{messages_text}")

        return "\n\n".join(sections).strip()


def create_honcho_memory_service(**kwargs) -> HonchoMemoryService:
    """Create and initialize the default Honcho memory service."""
    service = HonchoMemoryService(**kwargs)
    service.initialize()
    return service


def _messages_to_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages.strip()
    if not isinstance(messages, list):
        if messages is None:
            return ""
        return str(messages).strip()

    lines: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role") or "memory"
            content = message.get("content") or ""
        else:
            role = getattr(message, "role", None) or getattr(message, "peer_id", "memory")
            content = getattr(message, "content", "")
        content = str(content).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _peer_card_to_text(peer_card: Any) -> str:
    if peer_card is None:
        return ""
    if isinstance(peer_card, str):
        return peer_card.strip()
    if isinstance(peer_card, list):
        return "\n".join(f"- {str(item).strip()}" for item in peer_card if str(item).strip())
    return str(peer_card).strip()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default
