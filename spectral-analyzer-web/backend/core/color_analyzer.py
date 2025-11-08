"""
Grease Color Analyzer - FTIR Spectral Data to Color Visualization

Calculates grease color based on spectral features and chemical composition
identified in FTIR data. This provides visual representation of grease condition
for machinery health assessment.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def get_max_in_range(wavenumbers: List[float], absorbances: List[float], 
                     min_wn: float, max_wn: float) -> float:
    """
    Find maximum absorbance value within a specified wavenumber range.
    
    Args:
        wavenumbers: List of wavenumber values (cm⁻¹)
        absorbances: List of absorbance values
        min_wn: Minimum wavenumber for the range
        max_wn: Maximum wavenumber for the range
        
    Returns:
        Maximum absorbance value in the range, or 0.0 if range not found
    """
    try:
        # Convert to numpy arrays for efficient processing
        wn_array = np.array(wavenumbers)
        abs_array = np.array(absorbances)
        
        # Find indices within the range
        mask = (wn_array >= min_wn) & (wn_array <= max_wn)
        
        # Get absorbances in range
        values_in_range = abs_array[mask]
        
        if len(values_in_range) > 0:
            return float(np.max(values_in_range))
        else:
            return 0.0
            
    except Exception as e:
        logger.warning(f"Error finding max in range [{min_wn}, {max_wn}]: {e}")
        return 0.0


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hexadecimal color code.
    
    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)
        
    Returns:
        Hex color string in format #RRGGBB
    """
    # Clamp values to valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    return f"#{r:02X}{g:02X}{b:02X}"


def get_color_description(r: int, g: int, b: int) -> str:
    """
    Generate descriptive color name based on RGB values.
    
    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)
        
    Returns:
        Human-readable color description
    """
    # Calculate brightness
    brightness = (r + g + b) / 3
    
    # Determine darkness level
    if brightness < 60:
        darkness = "Very Dark"
    elif brightness < 100:
        darkness = "Dark"
    elif brightness < 150:
        darkness = "Medium"
    elif brightness < 200:
        darkness = "Light"
    else:
        darkness = "Very Light"
    
    # Determine color family based on dominant component
    max_component = max(r, g, b)
    
    # Amber/Brown range (typical for grease)
    if r > g > b:
        if r - g < 40 and g - b < 40:
            base_color = "Amber-Brown"
        elif r - g < 60:
            base_color = "Amber"
        else:
            base_color = "Brown-Red"
    elif r > b > g:
        base_color = "Brown"
    elif g > r and g > b:
        if r > b:
            base_color = "Yellow-Green"
        else:
            base_color = "Green"
    elif b > r and b > g:
        base_color = "Blue-Gray"
    else:
        # Similar RGB values
        if brightness < 100:
            base_color = "Black-Brown"
        elif brightness > 180:
            base_color = "Light Tan"
        else:
            base_color = "Gray-Brown"
    
    return f"{darkness} {base_color}"


def calculate_grease_color(wavenumbers: List[float], 
                          absorbances: List[float]) -> Dict:
    """
    Calculate grease color from FTIR spectral data.
    
    This algorithm analyzes key spectral features to determine grease color,
    which correlates with oxidation state, contamination, and overall condition.
    
    Args:
        wavenumbers: List of wavenumber values (cm⁻¹)
        absorbances: List of absorbance values
        
    Returns:
        Dictionary containing:
        - rgb: Dictionary with r, g, b values (0-255)
        - hex: Hexadecimal color code (#RRGGBB)
        - description: Human-readable color name
        - analysis: Dictionary with spectral analysis details
    """
    try:
        logger.info(f"Calculating grease color from {len(wavenumbers)} spectral points")
        
        # Step 1: Find max absorbance in specific ranges
        max_absorbance = float(np.max(absorbances)) if len(absorbances) > 0 else 0.0
        
        # C-H stretching region (2800-3000 cm⁻¹) - indicates hydrocarbon content
        ch_stretch_max = get_max_in_range(wavenumbers, absorbances, 2800, 3000)
        
        # CH₂ rocking region (700-750 cm⁻¹) - chain length indicator
        ch2_rocking_max = get_max_in_range(wavenumbers, absorbances, 700, 750)
        
        # C=O stretching region (1650-1750 cm⁻¹) - oxidation indicator
        carbonyl_max = get_max_in_range(wavenumbers, absorbances, 1650, 1750)
        
        # Step 2: Calculate darkness factor
        # Higher absorbance = darker color
        darkness_factor = min(max_absorbance / 2.0, 1.0)
        
        # Step 3: Base amber color for petroleum grease
        # This is the typical color of fresh grease
        base_r, base_g, base_b = 180, 120, 50
        
        # Step 4: Apply darkness based on total absorbance
        # Higher absorbance indicates more material/thicker sample/darker color
        r = int(base_r * (1 - darkness_factor * 0.6))
        g = int(base_g * (1 - darkness_factor * 0.65))
        b = int(base_b * (1 - darkness_factor * 0.7))
        
        # Step 5: Adjust for oxidation
        # High carbonyl content indicates oxidation, which adds reddish tint
        if carbonyl_max > 0.5:
            oxidation_factor = min(carbonyl_max / 2.0, 0.3)
            r = min(255, int(r + 15 * (1 + oxidation_factor)))
            # Slightly reduce green/blue for more brown appearance
            g = max(0, int(g * (1 - oxidation_factor * 0.2)))
            b = max(0, int(b * (1 - oxidation_factor * 0.3)))
        
        # Step 6: Adjust for degradation
        # Low C-H content relative to other peaks suggests severe degradation
        if max_absorbance > 0 and ch_stretch_max / max_absorbance < 0.3:
            # More degraded = darker, more brown
            degradation_factor = 0.3
            r = max(0, int(r * (1 - degradation_factor * 0.3)))
            g = max(0, int(g * (1 - degradation_factor * 0.4)))
            b = max(0, int(b * (1 - degradation_factor * 0.5)))
        
        # Ensure values are in valid range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        # Generate hex code and description
        hex_code = rgb_to_hex(r, g, b)
        description = get_color_description(r, g, b)
        
        # Prepare analysis explanation
        analysis_notes = []
        
        if darkness_factor > 0.7:
            analysis_notes.append("High absorbance indicates concentrated sample or thick grease layer")
        elif darkness_factor < 0.3:
            analysis_notes.append("Low absorbance suggests diluted or thin sample")
        
        if carbonyl_max > 0.5:
            analysis_notes.append("Significant carbonyl peak detected - oxidation present")
        elif carbonyl_max > 0.2:
            analysis_notes.append("Moderate carbonyl peak - some oxidation detected")
        else:
            analysis_notes.append("Minimal carbonyl peak - low oxidation")
        
        if ch_stretch_max > 1.0:
            analysis_notes.append("Strong C-H stretching indicates good hydrocarbon content")
        elif ch_stretch_max < 0.5:
            analysis_notes.append("Weak C-H stretching may indicate degradation or contamination")
        
        result = {
            'rgb': {
                'r': r,
                'g': g,
                'b': b
            },
            'hex': hex_code,
            'description': description,
            'analysis': {
                'ch_stretch_max': round(ch_stretch_max, 4),
                'ch2_rocking_max': round(ch2_rocking_max, 4),
                'carbonyl_max': round(carbonyl_max, 4),
                'max_absorbance': round(max_absorbance, 4),
                'darkness_factor': round(darkness_factor, 4),
                'spectral_features': {
                    'hydrocarbon_content': 'High' if ch_stretch_max > 1.0 else 'Moderate' if ch_stretch_max > 0.5 else 'Low',
                    'oxidation_level': 'High' if carbonyl_max > 0.5 else 'Moderate' if carbonyl_max > 0.2 else 'Low',
                    'sample_concentration': 'High' if darkness_factor > 0.7 else 'Moderate' if darkness_factor > 0.4 else 'Low'
                },
                'notes': analysis_notes
            }
        }
        
        logger.info(f"Color calculation successful: {hex_code} ({description})")
        return result
        
    except Exception as e:
        logger.error(f"Color calculation failed: {e}", exc_info=True)
        # Return default color on error
        return {
            'rgb': {'r': 150, 'g': 100, 'b': 50},
            'hex': '#966432',
            'description': 'Medium Amber (Error in calculation)',
            'analysis': {
                'ch_stretch_max': 0.0,
                'ch2_rocking_max': 0.0,
                'carbonyl_max': 0.0,
                'max_absorbance': 0.0,
                'darkness_factor': 0.0,
                'spectral_features': {
                    'hydrocarbon_content': 'Unknown',
                    'oxidation_level': 'Unknown',
                    'sample_concentration': 'Unknown'
                },
                'notes': [f'Calculation error: {str(e)}']
            }
        }