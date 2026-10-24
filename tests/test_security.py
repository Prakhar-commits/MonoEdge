from app.security import signature_is_valid
from tests.conftest import sign

BODY = b'{"event_id":"e1","type":"ticket.created","data":{}}'
SECRET = "unit-test-secret"


def test_accepts_plain_hex_and_sha256_prefix():
    good = sign(BODY, SECRET)
    assert signature_is_valid(BODY, good, SECRET)
    assert signature_is_valid(BODY, f"sha256={good}", SECRET)
    assert signature_is_valid(BODY, f"  sha256={good}  ", SECRET)  # surrounding spaces


def test_rejects_wrong_missing_or_empty_signatures():
    good = sign(BODY, SECRET)
    assert not signature_is_valid(BODY, None, SECRET)
    assert not signature_is_valid(BODY, "", SECRET)
    assert not signature_is_valid(BODY, "sha256=", SECRET)
    assert not signature_is_valid(BODY, good[:-1], SECRET)  # truncated
    assert not signature_is_valid(BODY, good.upper(), SECRET)  # hex digest is lower-case
    assert not signature_is_valid(BODY, sign(BODY, "another-secret"), SECRET)


def test_rejects_when_body_changes_by_one_byte():
    good = sign(BODY, SECRET)
    assert not signature_is_valid(BODY + b" ", good, SECRET)


def test_non_ascii_signature_is_rejected_not_a_crash():
    # hmac.compare_digest raises TypeError for non-ASCII str, so we compare bytes instead.
    assert not signature_is_valid(BODY, "sha256=ééé", SECRET)
