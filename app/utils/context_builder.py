def build_campaign_context(project: dict) -> str:
    """
    Parses the Redis project dictionary and builds a comprehensive LLM context string.
    Expects project['config_json'] to be a list of dicts, e.g. [{"type": "2 BHK", "area": "1200 sqft", "price": "1.2 Cr"}]
    """
    context_lines = []
    
    context_lines.append(f"Project Name: {project.get('name')}")
    context_lines.append(f"Location: {project.get('locality')}")
    
    min_price = project.get('min_price')
    max_price = project.get('max_price')
    if min_price and max_price:
        context_lines.append(f"Overall Price Range: {min_price} to {max_price} Lakhs")
    
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
