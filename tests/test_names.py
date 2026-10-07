from aml.names import make_aliases


def test_aliases_unique_stable_and_readable():
    ids = [f"{b:03d}_{a}" for b in range(20) for a in range(1000, 1400)]
    a1 = make_aliases(ids)
    assert a1 == make_aliases(reversed(ids))
    assert len(set(a1.values())) == len(ids)
    name = a1[ids[0]]
    assert len(name.split()) == 3 and name.split()[2].isdigit()
