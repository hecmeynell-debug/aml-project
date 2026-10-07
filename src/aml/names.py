"""Readable, stable display names for account IDs, e.g. ``070_100428660`` -> ``Amber Falcon 417``.

Names are derived from a hash of the account ID, so they carry no information about the account
(no risk or label leakage) and are the same on every run for the same set of accounts. Collisions
are resolved by re-hashing with a salt, processing IDs in sorted order.
"""
from __future__ import annotations

from hashlib import blake2b

ADJECTIVES = (
    "Amber Azure Bold Brave Bright Calm Clever Coral Crimson Crisp Dusty Eager Early Fancy Fierce Gentle "
    "Golden Grand Green Hazel Humble Icy Indigo Ivory Jade Jolly Keen Kind Lively Lucky Lunar Maple "
    "Mellow Misty Noble Olive Opal Pale Plum Proud Quick Quiet Rapid Rosy Royal Ruby Rustic Sandy "
    "Scarlet Shy Silent Silver Sleek Solar Steady Stormy Sunny Swift Tidy Velvet Violet Witty Zesty Zen"
).split()
NOUNS = (
    "Badger Bear Beaver Bison Condor Cougar Crane Crow Deer Dingo Dolphin Dove Eagle Egret Falcon Ferret "
    "Finch Fox Gecko Gull Hare Hawk Heron Ibis Jaguar Kestrel Kite Koala Lemur Lion Lynx Magpie "
    "Marten Mink Moose Newt Orca Osprey Otter Owl Panda Parrot Pelican Puffin Quail Raven Robin Seal "
    "Shark Sparrow Stoat Swan Tapir Tiger Toucan Turtle Viper Walrus Whale Wolf Wombat Wren Yak Zebra"
).split()
assert len(ADJECTIVES) == len(NOUNS) == 64


def _name(account_id: str, salt: int) -> str:
    h = int.from_bytes(blake2b(f"{salt}:{account_id}".encode(), digest_size=8).digest(), "big")
    return f"{ADJECTIVES[h % 64]} {NOUNS[(h >> 6) % 64]} {(h >> 12) % 1000:03d}"


def make_aliases(account_ids) -> dict[str, str]:
    """Map every account ID to a unique readable name."""
    out: dict[str, str] = {}
    used: set[str] = set()
    for acc in sorted(set(account_ids)):
        salt = 0
        name = _name(acc, salt)
        while name in used:
            salt += 1
            name = _name(acc, salt)
        used.add(name)
        out[acc] = name
    return out
