import pytest
from app.core import router, providers
import pytest_asyncio
import asyncio

# Mock Provider
class MockProvider(providers.BaseProvider):
    def __init__(self, name, should_fail=False):
        super().__init__(name, ["*"], {})
        self.should_fail = should_fail
        self.call_count = 0

    async def stream_chat(self, model, messages):
        self.call_count += 1
        if self.should_fail:
            raise Exception("Mock Fail")
        yield "Mock Response"

@pytest.mark.asyncio
async def test_router_fallback():
    # Setup Router with 1 bad provider and 1 good provider
    r = router.ProviderRouter()
    
    bad_provider = MockProvider("bad", should_fail=True)
    good_provider = MockProvider("good", should_fail=False)
    
    # Overwrite loaded providers for testing
    r.providers = [bad_provider, good_provider]
    # start at 1 so next is 0 (bad)
    r.current_index = 1
    
    # Run stream
    chunks = []
    async for chunk in r.stream_chat("test-model", [{"role":"user", "content":"hi"}]):
        chunks.append(chunk)
        
    # Verify fallback happened
    assert len(chunks) == 1
    assert chunks[0] == "Mock Response"
    assert bad_provider.call_count == 1 # Tried once
    assert good_provider.call_count == 1 # Succeeded
    assert bad_provider.failure_count == 1
    assert good_provider.failure_count == 0

@pytest.mark.asyncio
async def test_router_all_fail():
    r = router.ProviderRouter()
    r.providers = [MockProvider("bad1", True), MockProvider("bad2", True)]
    
    with pytest.raises(Exception) as exc:
        async for _ in r.stream_chat("m", []):
            pass
            
    assert "All retries failed" in str(exc.value)
