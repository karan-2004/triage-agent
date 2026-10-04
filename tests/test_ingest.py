from triage.ingest import build_dev_profiles
from triage.ticket import Message, Thread


def test_build_dev_profiles_only_devs_and_lowercased_keys():
    t = Thread(id="101", messages=[
        Message("Alice", "fixed the login"),
        Message("support", "customer reported it"),
        Message("Customer", "it is broken"),
        Message("alice", "also patched the cache"),
    ])
    profiles = build_dev_profiles([t])
    assert set(profiles.keys()) == {"alice"}
    assert "fixed the login" in profiles["alice"]
    assert "also patched the cache" in profiles["alice"]
    assert "customer reported it" not in profiles["alice"]
    assert "it is broken" not in profiles["alice"]


def test_build_dev_profiles_empty_for_support_only():
    t = Thread(id="102", messages=[Message("support", "nothing to see")])
    assert build_dev_profiles([t]) == {}
