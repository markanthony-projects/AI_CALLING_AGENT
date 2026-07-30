import re
from typing import Optional

_LAKHS_PER_CRORE = 100.0
_PRICE_TOKEN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>cr|crore|crores|lakh|lakhs|l)\b", re.I)


def _price_to_crores(raw) -> Optional[float]:
    """Normalise a price to Crores. Bare numbers are Lakhs, matching the DB columns."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) / _LAKHS_PER_CRORE
    match = _PRICE_TOKEN.search(str(raw))
    if not match:
        return None
    value = float(match.group("value"))
    return value if match.group("unit").lower().startswith("cr") else value / _LAKHS_PER_CRORE


def _format_crores(value: float) -> str:
    return f"{f'{value:.2f}'.rstrip('0').rstrip('.')} Crores"


def _price_range(project: dict) -> Optional[str]:
    """One range, one unit, covering every price the agent can quote.

    min_price/max_price are Lakhs while config_json quotes Crores. Emitting both units left
    the agent to reconcile them mid-call, and it announced "1 to 2 Crores" for a project
    whose units run 1.2 to 3.5 Crores.
    """
    config = project.get("config_json")
    bounds = []
    if isinstance(config, list):
        bounds += [
            c
            for c in (_price_to_crores(i.get("price")) for i in config if isinstance(i, dict))
            if c is not None
        ]
    bounds += [
        c
        for c in (_price_to_crores(project.get("min_price")), _price_to_crores(project.get("max_price")))
        if c is not None
    ]
    if not bounds:
        return None
    return f"{_format_crores(min(bounds))} to {_format_crores(max(bounds))}"


def build_campaign_context(project: dict) -> str:
    """
    Parses the Redis project dictionary and builds a comprehensive LLM context string.
    Expects project['config_json'] to be a list of dicts, e.g. [{"type": "2 BHK", "area": "1200 sqft", "price": "1.2 Cr"}]
    """
    context_lines = []
    
    context_lines.append(f"Project Name: {project.get('name')}")
    context_lines.append(f"Location: {project.get('locality')}")
    
    price_range = _price_range(project)
    if price_range:
        context_lines.append(f"Overall Price Range: {price_range}")
    
    possession = project.get('possession_status')
    if possession:
        context_lines.append(f"Possession Status: {possession}")
        
    amenities = project.get('amenities')
    if amenities:
        context_lines.append(f"Amenities: {', '.join(amenities)}")
        
    usps = project.get('usps')
    if usps:
        context_lines.append(f"Key Selling Points (USPs): {', '.join(usps)}")
        
    nearby = project.get('nearby_facilities')
    if nearby and isinstance(nearby, dict):
        nearby_strs = [f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in nearby.items()]
        context_lines.append(f"Nearby Facilities: {' | '.join(nearby_strs)}")
        
    config = project.get('config_json')
    if config and isinstance(config, list):
        context_lines.append("Available Configurations (Unit Types, Area, and Pricing):")
        for item in config:
            if isinstance(item, dict):
                type_name = item.get("type", "Unit")
                area = item.get("area", "N/A area")
                price = item.get("price", "N/A price")
                context_lines.append(f"- {type_name}: {area}, Price: {price}")
            
    return "\n".join(context_lines)
