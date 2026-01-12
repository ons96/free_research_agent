import pytest
from app.core import shopping, search

def test_extract_price():
    assert shopping.extract_price("Price is $10.99") == 10.99
    assert shopping.extract_price("£50 deal") == 50.0
    assert shopping.extract_price("No price here") is None

def test_normalize_units():
    assert shopping.normalize_units(1, "kg") == 1000
    assert shopping.normalize_units(100, "g") == 100
    assert shopping.normalize_units(1, "lb") == 453.592

def test_analyze_deals():
    mock_results = [
        {"title": "Protein Powder 1kg", "body": "Best protein for $30.00 only!", "href": "http://test.com/1"},
        {"title": "Protein Powder 500g", "body": "Small pack $20.00", "href": "http://test.com/2"},
    ]
    deals = shopping.analyze_deals(mock_results)
    assert len(deals) == 2
    # Deal 1: $30 for 1000g => 0.03 $/g
    # Deal 2: $20 for 500g => 0.04 $/g
    assert deals[0].name == "Protein Powder 1kg"
    assert deals[0].price_per_unit < deals[1].price_per_unit

@pytest.mark.asyncio
async def test_search_mock(mocker):
    # Integration test with mock
    mock_ddg = mocker.patch("app.core.search.DDGS")
    mock_ctx = mock_ddg.return_value.__enter__.return_value
    mock_ctx.text.return_value = [{"title": "Test Result", "href": "http://test.com", "body": "Test Body"}]
    
    results = await search.search_web("testing")
    assert len(results) == 1
    assert results[0]["title"] == "Test Result"
