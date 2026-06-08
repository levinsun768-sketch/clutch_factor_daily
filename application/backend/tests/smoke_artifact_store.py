from app.core.config import load_settings
from app.services.artifact_store import ArtifactStore


def test_artifact_store_smoke():
    store = ArtifactStore(load_settings())
    health = store.health()
    assert health["ok"]
    assert store.latest_business_date()
    overview = store.overview(None, "all")
    assert overview["date"]
    assert "portfolio" in overview
    factors = store.factor_list(None, "all", "rankic", None, 5)
    assert factors
