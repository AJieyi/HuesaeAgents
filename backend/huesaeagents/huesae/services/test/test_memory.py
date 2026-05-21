"""Honcho memory service tests."""

from __future__ import annotations

from pathlib import Path

from huesaeagents.huesae.services.memory import HonchoMemoryService


class _FakeConfig:
    def __init__(self, **kwargs):
        self.values = kwargs


class _FakePeer:
    def __init__(self, peer_id: str):
        self.id = peer_id
        self.configuration = None

    def message(self, content: str):
        return {"peer_id": self.id, "role": self.id, "content": content}

    def set_configuration(self, configuration):
        self.configuration = configuration


class _FakeContext:
    def __init__(self, text: str):
        self._text = text

    def to_openai(self):
        return [{"role": "system", "content": self._text}]


class _FakeSession:
    def __init__(self, session_id: str):
        self.id = session_id
        self.messages = []
        self.peers = []
        self.context_calls = []

    def add_peers(self, peers):
        self.peers.extend(peers)

    def add_messages(self, messages):
        self.messages.extend(messages)

    def context(self, **kwargs):
        self.context_calls.append(kwargs)
        return _FakeContext("记忆：用户喜欢猫")


class _FakeHoncho:
    def __init__(self):
        self.sessions = {}
        self.peers = {}

    def peer(self, peer_id: str):
        peer = self.peers.get(peer_id)
        if peer is None:
            peer = _FakePeer(peer_id)
            self.peers[peer_id] = peer
        return peer

    def session(self, session_id: str):
        session = self.sessions.get(session_id)
        if session is None:
            session = _FakeSession(session_id)
            self.sessions[session_id] = session
        return session


def test_memory_service_persists_session_id(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "session.json"
    client = _FakeHoncho()

    service = HonchoMemoryService(
        client=client,
        base_url="http://localhost:8000",
        workspace_id="demo-workspace",
        state_path=state_path,
    )
    monkeypatch.setattr(service, "_check_server", lambda: None)
    monkeypatch.setattr(
        service,
        "_session_peers_with_config",
        lambda: [
            (service.user_peer, _FakeConfig(observe_me=True, observe_others=False)),
            (service.assistant_peer, _FakeConfig(observe_me=False, observe_others=True)),
        ],
    )

    assert service.initialize() is True
    assert service.enabled is True
    assert service.session_id == "demo-workspace_local-user"
    assert state_path.exists()

    assert service.store_exchange("我喜欢猫", "好呀，我记住了") is True
    assert len(service.session.messages) == 2
    assert service.session.messages[0]["content"] == "我喜欢猫"
    assert service.session.messages[1]["content"] == "好呀，我记住了"
    assert service.session.peers[0][1].values["observe_me"] is True
    assert service.session.peers[1][1].values["observe_me"] is False


def test_memory_service_returns_prompt_text_and_searches_by_user_input(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "session.json"
    client = _FakeHoncho()

    service = HonchoMemoryService(
        client=client,
        base_url="http://localhost:8000",
        workspace_id="demo-workspace",
        state_path=state_path,
    )
    monkeypatch.setattr(service, "_check_server", lambda: None)

    assert service.initialize() is True
    text = service.get_context(user_input="我喜欢什么动物？")

    assert "用户喜欢猫" in text
    call = service.session.context_calls[-1]
    assert call["summary"] is True
    assert call["peer_target"] == "local-user"
    assert call["search_query"] == "我喜欢什么动物？"
    assert call["include_most_frequent"] is True


def test_memory_service_restores_existing_session_id(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "session.json"
    state_path.write_text(
        (
            '{\n'
            '  "workspace_id": "demo-workspace",\n'
            '  "user_peer_id": "local-user",\n'
            '  "assistant_peer_id": "huesae-main-agent",\n'
            '  "session_id": "existing-session"\n'
            '}\n'
        ),
        encoding="utf-8",
    )
    client = _FakeHoncho()

    service = HonchoMemoryService(
        client=client,
        base_url="http://localhost:8000",
        workspace_id="demo-workspace",
        state_path=state_path,
    )
    monkeypatch.setattr(service, "_check_server", lambda: None)

    assert service.initialize() is True
    assert service.session_id == "existing-session"
    assert service.session.id == "existing-session"
