from env.generate_world import World, hash_tree


def test_same_seed_same_bytes(tmp_path):
    World(42).build().write(tmp_path / "a")
    World(42).build().write(tmp_path / "b")
    assert hash_tree(tmp_path / "a") == hash_tree(tmp_path / "b")


def test_different_seed_different_world(tmp_path):
    World(42).build().write(tmp_path / "a")
    World(7).build().write(tmp_path / "b")
    assert hash_tree(tmp_path / "a") != hash_tree(tmp_path / "b")


def test_world_has_expected_files(tmp_path):
    World(42).build().write(tmp_path)
    assert len(list((tmp_path / "logs").rglob("*.log"))) == 8 * 5  # 8 services x 5 days
    for name in ["deploys.csv", "oncall.yaml", "README.md"]:
        assert (tmp_path / name).exists()
    assert len(list((tmp_path / "runbooks").glob("*.md"))) == 6