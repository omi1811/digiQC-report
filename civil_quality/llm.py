"""
Civil Quality Expert - Self-contained Knowledge Engine
No external LLM required. Uses comprehensive civil engineering knowledge base.
Designed to run on Render free tier without external backends.
"""

import re
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Civil Quality AI assistant.
Your job is to help site engineers with practical civil engineering and quality control guidance."""

# ==================== COMPREHENSIVE KNOWLEDGE BASE ====================

# IS Code References for Civil Quality
IS_CODES = {
    "IS 456:2000": {
        "title": "Plain and Reinforced Concrete - Code of Practice",
        "topics": ["concrete", "reinforcement", "cover", "curing", "formwork", "mix design"],
        "key_provisions": {
            "minimum_cover": {
                "mild": {"slab": 20, "beam": 25, "column": 40, "footing": 50},
                "moderate": {"slab": 30, "beam": 30, "column": 40, "footing": 50},
                "severe": {"slab": 45, "beam": 45, "column": 45, "footing": 75},
                "very_severe": {"slab": 50, "beam": 50, "column": 50, "footing": 75},
                "extreme": {"slab": 75, "beam": 75, "column": 75, "footing": 75}
            },
            "minimum_grade": {
                "mild": "M20",
                "moderate": "M25", 
                "severe": "M30",
                "very_severe": "M35",
                "extreme": "M40"
            },
            "curing_period": {
                "OPC": 7,
                "PPC": 10,
                "PSC": 14
            },
            "max_water_cement_ratio": {
                "mild": 0.55,
                "moderate": 0.50,
                "severe": 0.45,
                "very_severe": 0.45,
                "extreme": 0.40
            }
        }
    },
    "IS 1786:2008": {
        "title": "High Strength Deformed Steel Bars for Concrete Reinforcement",
        "topics": ["rebar", "steel", "reinforcement", "TMT"],
        "key_provisions": {
            "grades": ["Fe 415", "Fe 500", "Fe 500D", "Fe 550", "Fe 550D", "Fe 600"],
            "bend_test": "Should not crack when bent around mandrel",
            "elongation": {"Fe 415": 14.5, "Fe 500": 12, "Fe 500D": 16, "Fe 550": 10, "Fe 550D": 14.5}
        }
    },
    "IS 10262:2019": {
        "title": "Concrete Mix Proportioning - Guidelines",
        "topics": ["mix design", "concrete", "proportioning"],
        "key_provisions": {
            "slump_values": {
                "mass_concrete": "25-50mm",
                "normal_RCC": "50-100mm",
                "pumped_concrete": "75-100mm",
                "trench_fill": "100-150mm"
            }
        }
    },
    "IS 2502:1963": {
        "title": "Code of Practice for Bending and Fixing of Bars",
        "topics": ["bending", "reinforcement", "bar fixing", "lap length"],
        "key_provisions": {
            "minimum_bend_dia": "4 times bar diameter for stirrups, 5 times for main bars",
            "lap_length": "Generally 50 times bar diameter in tension zones"
        }
    },
    "IS 14687:1999": {
        "title": "Quality Assurance during Construction of Buildings",
        "topics": ["quality", "construction", "inspection", "checklist"],
        "key_provisions": {
            "inspection_stages": ["Before concreting", "During concreting", "After concreting"],
            "records_required": ["Cube test results", "Steel test certificates", "Mix design approval"]
        }
    }
}

# Common Quality Issues and Solutions
QUALITY_ISSUES = {
    "honeycomb": {
        "description": "Voids in concrete due to improper compaction",
        "causes": ["Poor vibration", "Congested reinforcement", "Improper cover", "Low workability"],
        "prevention": ["Use proper vibrator", "Ensure minimum cover", "Use plasticizer if needed", "Follow layer-by-layer concreting"],
        "remediation": ["Chip out loose material", "Apply bonding agent", "Fill with non-shrink grout", "Document and get approval"]
    },
    "cold_joint": {
        "description": "Discontinuity in concrete due to delay between pours",
        "causes": ["Delay > initial setting time", "Improper planning", "Equipment breakdown"],
        "prevention": ["Pour within 30 mins", "Use retarder if delay expected", "Keep surface wet"],
        "remediation": ["Inject epoxy grout", "Apply bonding chemical before next pour"]
    },
    "segregation": {
        "description": "Separation of coarse aggregate from mortar",
        "causes": ["Excess free fall height", "Over-vibration", "High slump"],
        "prevention": ["Limit free fall to 1.5m", "Use tremie pipe", "Control slump"],
        "remediation": ["Remove and recast if severe", "Apply repair mortar for surface issues"]
    },
    "bleeding": {
        "description": "Water rising to surface of fresh concrete",
        "causes": ["High water content", "Fine aggregate deficiency", "Over-vibration"],
        "prevention": ["Proper mix design", "Adequate fines", "Controlled vibration"],
        "remediation": ["Remove bleed water before finishing", "Re-vibrate if excessive"]
    },
    "cracks": {
        "description": "Various types of cracks in concrete",
        "types": {
            "plastic_shrinkage": {"cause": "Rapid evaporation", "prevention": "Cover, mist spray, wind barrier"},
            "drying_shrinkage": {"cause": "Moisture loss", "prevention": "Proper curing"},
            "thermal": {"cause": "Heat of hydration", "prevention": "Use low-heat cement, cooling pipes"},
            "structural": {"cause": "Overload/design issue", "prevention": "Verify design, check loads"}
        }
    }
}

# Checklist Templates
CHECKLISTS = {
    "pre_concreting": {
        "title": "Pre-Concreting Checklist (As per IS 456:2000)",
        "items": [
            "Formwork cleaned and oiled",
            "Cover blocks placed at correct spacing (1m c/c for slab)",
            "Reinforcement checked against drawing",
            "Lap length verified (50d min in tension)",
            "Chair bars provided for top reinforcement",
            "Starter bars/dowels checked",
            "Embedments/openings as per drawing",
            "Formwork alignment and level checked",
            "Cube moulds prepared and labeled",
            "Mix design approved and available",
            "Concreting sequence planned",
            "Weather forecast checked"
        ]
    },
    "during_concreting": {
        "title": "During Concreting Checklist",
        "items": [
            "Slump test conducted (record value)",
            "Pouring height < 1.5m (use tremie if deeper)",
            "Layer thickness < 450mm",
            "Vibration time 10-15 seconds per point",
            "No displacement of reinforcement",
            "Cold joints avoided (pour within 30 mins)",
            "Cubes cast and labeled properly",
            "Ambient temperature recorded"
        ]
    },
    "post_concreting": {
        "title": "Post-Concreting Checklist",
        "items": [
            "Curing started within 24 hours",
            "Curing method: ponding/spraying/curing compound",
            "Curing period: 7 days OPC, 10 days PPC",
            "Formwork removal as per schedule",
            "Surface defects documented",
            "Cube testing at 7 and 28 days",
            "Deshuttering props if required"
        ]
    },
    "steel_inspection": {
        "title": "Steel/Reinforcement Inspection Checklist",
        "items": [
            "Mill test certificate verified",
            "Physical verification: grade marking visible",
            "No rust scaling (light rust acceptable)",
            "Bar diameter within tolerance",
            "Bending schedule checked",
            "Minimum bend diameter verified",
            "Binding wire 18 gauge black annealed",
            "Storage: off ground, under cover"
        ]
    },
    "formwork": {
        "title": "Formwork Inspection Checklist",
        "items": [
            "Formwork design/drawing available",
            "Props spacing as per design",
            "Wedges/jacks properly tightened",
            "Joints sealed (no grout leakage)",
            "Camber provided where required",
            "Release agent applied",
            "Scaffolding/working platform safe",
            "Stripping schedule displayed"
        ]
    }
}

# Material Specifications
MATERIAL_SPECS = {
    "cement": {
        "types": {
            "OPC 33": "General construction, initial strength critical",
            "OPC 43": "General RCC work, standard construction",
            "OPC 53": "High strength concrete, precast",
            "PPC": "Mass concrete, sulphate resistance, reduced heat",
            "PSC": "Marine works, sulphate resistance",
            "SRC": "High sulphate exposure"
        },
        "storage": "Off ground on platform, max 10 bags high, FIFO usage",
        "shelf_life": "3 months after manufacturing (check date)",
        "tests": ["Fineness", "Soundness", "Setting time", "Compressive strength"]
    },
    "aggregates": {
        "coarse": {
            "size": "20mm for general RCC, 40mm for mass concrete",
            "grading": "As per IS 383",
            "shape": "Angular/cubical preferred, flakiness < 25%",
            "tests": ["Sieve analysis", "Flakiness", "Impact value", "Crushing value"]
        },
        "fine": {
            "zones": "Zone I (Coarse) to Zone IV (Fine) - Zone II/III preferred",
            "FM": "2.6-2.9 generally preferred",
            "silt": "< 3% for concreting work",
            "tests": ["Sieve analysis", "Silt content", "Bulking"]
        }
    },
    "water": {
        "quality": "Potable water preferred, pH 6-8",
        "limits": {
            "sulphates": "< 400 mg/l",
            "chlorides": "< 500 mg/l for RCC, < 2000 for PCC",
            "suspended_matter": "< 2000 mg/l"
        }
    }
}


class CivilQualityExpert:
    """Self-contained Civil Quality Expert using comprehensive knowledge base."""
    
    def __init__(self):
        self.knowledge_base = {
            "is_codes": IS_CODES,
            "quality_issues": QUALITY_ISSUES,
            "checklists": CHECKLISTS,
            "materials": MATERIAL_SPECS
        }
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response using rule-based logic and knowledge base.
        No external API required.
        """
        try:
            # Parse the user prompt to extract context and question
            context, question = self._parse_prompt(user_prompt)
            
            # Generate intelligent response
            response = self._generate_response(question, context)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(user_prompt)
    
    def _parse_prompt(self, prompt: str) -> tuple:
        """Parse the structured prompt to extract context and question."""
        context = {}
        question = ""
        
        # Extract PROJECT_INFO section
        if "PROJECT_INFO:" in prompt:
            try:
                start = prompt.find("PROJECT_INFO:") + len("PROJECT_INFO:")
                end = prompt.find("WEB_CONTEXT:")
                if end == -1:
                    end = prompt.find("USER_QUESTION:")
                info_str = prompt[start:end].strip()
                context = json.loads(info_str)
            except:
                pass
        
        # Extract USER_QUESTION section
        if "USER_QUESTION:" in prompt:
            start = prompt.find("USER_QUESTION:") + len("USER_QUESTION:")
            end = prompt.find("Now:", start)
            if end == -1:
                end = len(prompt)
            question = prompt[start:end].strip()
        else:
            question = prompt
        
        return context, question
    
    def _generate_response(self, question: str, context: dict) -> str:
        """Generate intelligent response based on question and context."""
        q_lower = question.lower()
        
        # Detect intent
        if any(word in q_lower for word in ["checklist", "check", "verify", "inspection"]):
            return self._handle_checklist_query(question, context)
        
        elif any(word in q_lower for word in ["cover", "clear cover", "nominal cover"]):
            return self._handle_cover_query(question, context)
        
        elif any(word in q_lower for word in ["grade", "m20", "m25", "m30", "m35", "m40", "suitable", "okay"]):
            return self._handle_grade_query(question, context)
        
        elif any(word in q_lower for word in ["curing", "cure"]):
            return self._handle_curing_query(question, context)
        
        elif any(word in q_lower for word in ["honeycomb", "cold joint", "crack", "segregation", "bleeding", "defect"]):
            return self._handle_defect_query(question, context)
        
        elif any(word in q_lower for word in ["lap", "splice", "development length"]):
            return self._handle_lap_length_query(question, context)
        
        elif any(word in q_lower for word in ["slump", "workability"]):
            return self._handle_slump_query(question, context)
        
        elif any(word in q_lower for word in ["cement", "opc", "ppc", "psc"]):
            return self._handle_cement_query(question, context)
        
        elif any(word in q_lower for word in ["aggregate", "sand", "coarse", "fine"]):
            return self._handle_aggregate_query(question, context)
        
        elif any(word in q_lower for word in ["water cement ratio", "w/c ratio", "wcr"]):
            return self._handle_wcr_query(question, context)
        
        elif any(word in q_lower for word in ["formwork", "shuttering", "deshuttering"]):
            return self._handle_formwork_query(question, context)
        
        elif any(word in q_lower for word in ["rebar", "reinforcement", "steel", "bar"]):
            return self._handle_steel_query(question, context)
        
        elif any(word in q_lower for word in ["test", "cube", "cylinder", "sample"]):
            return self._handle_testing_query(question, context)
        
        else:
            return self._handle_general_query(question, context)
    
    def _handle_checklist_query(self, question: str, context: dict) -> str:
        """Handle checklist-related queries."""
        q_lower = question.lower()
        
        if "pre" in q_lower or "before" in q_lower:
            checklist = CHECKLISTS["pre_concreting"]
        elif "during" in q_lower:
            checklist = CHECKLISTS["during_concreting"]
        elif "post" in q_lower or "after" in q_lower:
            checklist = CHECKLISTS["post_concreting"]
        elif "steel" in q_lower or "rebar" in q_lower or "reinforcement" in q_lower:
            checklist = CHECKLISTS["steel_inspection"]
        elif "formwork" in q_lower or "shutter" in q_lower:
            checklist = CHECKLISTS["formwork"]
        else:
            # Return all relevant checklists
            return self._format_all_checklists()
        
        return self._format_checklist(checklist)
    
    def _handle_cover_query(self, question: str, context: dict) -> str:
        """Handle cover-related queries."""
        exposure = context.get("exposure_condition", "").lower() or "moderate"
        member = context.get("member_type", "").lower() or "slab"
        
        cover_data = IS_CODES["IS 456:2000"]["key_provisions"]["minimum_cover"]
        
        # Normalize exposure
        exposure_key = exposure.replace(" ", "_")
        if exposure_key not in cover_data:
            exposure_key = "moderate"
        
        covers = cover_data[exposure_key]
        
        response = f"""**Answer for Site Engineer:**

## Minimum Cover Requirements (IS 456:2000, Table 16)

For **{exposure.title()} exposure conditions**:

| Member | Minimum Cover (mm) |
|--------|-------------------|
| Slab | {covers.get('slab', 20)} |
| Beam | {covers.get('beam', 25)} |
| Column | {covers.get('column', 40)} |
| Footing | {covers.get('footing', 50)} |

**Your Query:** For **{member.title()}** in **{exposure.title()}** exposure → **{covers.get(member, 30)} mm** minimum cover.

**Practical Tips:**
- Use cover blocks at 1m c/c spacing for slabs
- Check cover at all corners and edges
- Account for construction tolerance (+5mm recommended)
- For bundled bars, measure from outermost bar

**Reference:** IS 456:2000, Clause 26.4.1, Table 16
"""
        return response
    
    def _handle_grade_query(self, question: str, context: dict) -> str:
        """Handle concrete grade suitability queries."""
        exposure = context.get("exposure_condition", "").lower() or "moderate"
        member = context.get("member_type", "").lower()
        grade = context.get("grade_of_concrete", "").upper()
        
        min_grades = IS_CODES["IS 456:2000"]["key_provisions"]["minimum_grade"]
        
        # Normalize exposure
        exposure_key = exposure.replace(" ", "_")
        if exposure_key not in min_grades:
            exposure_key = "moderate"
        
        min_required = min_grades[exposure_key]
        
        # Check suitability
        if grade:
            grade_num = int(re.search(r'\d+', grade).group()) if re.search(r'\d+', grade) else 0
            min_num = int(re.search(r'\d+', min_required).group()) if re.search(r'\d+', min_required) else 20
            
            is_suitable = grade_num >= min_num
            
            response = f"""**Answer for Site Engineer:**

## Concrete Grade Suitability Check

**Your Input:**
- Grade: **{grade}**
- Exposure: **{exposure.title()}**
- Member: **{member.title() if member else 'General RCC'}**

**Assessment:** {'✅ **SUITABLE**' if is_suitable else '❌ **NOT SUITABLE**'}

**Reasoning:**
- Minimum grade for {exposure.title()} exposure: **{min_required}**
- Your specified grade: **{grade}**
- {'Grade meets or exceeds minimum requirement.' if is_suitable else f'Grade is below minimum. Use at least {min_required}.'}

"""
        else:
            response = f"""**Answer for Site Engineer:**

## Minimum Concrete Grade Requirements (IS 456:2000, Table 5)

| Exposure Condition | Minimum Grade |
|-------------------|---------------|
| Mild | M20 |
| Moderate | M25 |
| Severe | M30 |
| Very Severe | M35 |
| Extreme | M40 |

**For {exposure.title()} exposure:** Minimum **{min_required}** required.

"""
        
        response += """**Additional Recommendations:**
- Always verify exposure conditions as per IS 456 Table 3
- Consider durability, not just strength
- Higher grade may be needed for structural requirements
- Document grade decision with proper justification

**Reference:** IS 456:2000, Clause 8.2.1, Table 5
"""
        return response
    
    def _handle_curing_query(self, question: str, context: dict) -> str:
        """Handle curing-related queries."""
        cement_type = context.get("cement_type", "").upper() or "OPC"
        
        curing_days = IS_CODES["IS 456:2000"]["key_provisions"]["curing_period"]
        
        response = f"""**Answer for Site Engineer:**

## Curing Requirements (IS 456:2000)

**Minimum Curing Period:**

| Cement Type | Minimum Days |
|-------------|-------------|
| OPC (43/53 grade) | 7 days |
| PPC (Pozzolana) | 10 days |
| PSC (Slag) | 14 days |

**For {cement_type}:** Minimum **{curing_days.get(cement_type, 7)} days** continuous curing.

**Curing Methods:**
1. **Ponding:** Best for slabs - maintain 50mm water depth
2. **Wet Covering:** Jute bags/hessian kept wet continuously
3. **Spraying:** Regular intervals, avoid letting surface dry
4. **Curing Compound:** Apply within 24 hours if water curing not possible

**Critical Points:**
- Start curing within 24 hours of casting
- Never let surface dry during curing period
- For hot weather: extend by 3-4 days
- For mass concrete: internal cooling may be needed

**Common Mistakes to Avoid:**
- ❌ Intermittent curing (worse than no curing)
- ❌ Early removal of formwork without curing
- ❌ Ignoring wind exposure

**Reference:** IS 456:2000, Clause 13.5
"""
        return response
    
    def _handle_defect_query(self, question: str, context: dict) -> str:
        """Handle concrete defect queries."""
        q_lower = question.lower()
        
        defect_key = None
        if "honeycomb" in q_lower:
            defect_key = "honeycomb"
        elif "cold joint" in q_lower:
            defect_key = "cold_joint"
        elif "segregation" in q_lower:
            defect_key = "segregation"
        elif "bleeding" in q_lower:
            defect_key = "bleeding"
        elif "crack" in q_lower:
            defect_key = "cracks"
        
        if defect_key and defect_key in QUALITY_ISSUES:
            defect = QUALITY_ISSUES[defect_key]
            
            response = f"""**Answer for Site Engineer:**

## {defect_key.replace('_', ' ').title()} - Complete Guide

**What is it?**
{defect['description']}

**Common Causes:**
"""
            for cause in defect.get('causes', []):
                response += f"- {cause}\n"
            
            response += "\n**Prevention Measures:**\n"
            for prev in defect.get('prevention', []):
                response += f"- ✅ {prev}\n"
            
            response += "\n**Remediation Steps:**\n"
            for i, rem in enumerate(defect.get('remediation', []), 1):
                response += f"{i}. {rem}\n"
            
        else:
            # General defects overview
            response = """**Answer for Site Engineer:**

## Common Concrete Defects - Quick Reference

| Defect | Main Cause | Key Prevention |
|--------|-----------|----------------|
| Honeycomb | Poor vibration | Proper compaction |
| Cold Joint | Delay between pours | Pour within 30 mins |
| Segregation | High free fall | Use tremie, limit drop height |
| Bleeding | Excess water | Control W/C ratio |
| Cracks | Various | Proper curing + design |

**Action Required:**
- Document defect with photos
- Mark affected area
- Get structural engineer's opinion
- Follow approved repair procedure
- Maintain records

"""
        
        response += "\n**Reference:** IS 456:2000, IS 14687:1999"
        return response
    
    def _handle_lap_length_query(self, question: str, context: dict) -> str:
        """Handle lap length and splice queries."""
        response = """**Answer for Site Engineer:**

## Lap Length / Development Length (IS 456:2000)

**Quick Formula:**
- **Tension Zone:** Lap = 50 × bar diameter (50d)
- **Compression Zone:** Lap = 40 × bar diameter (40d)

**Example Calculations:**

| Bar Dia | Tension Lap (50d) | Compression Lap (40d) |
|---------|-------------------|----------------------|
| 8 mm | 400 mm | 320 mm |
| 10 mm | 500 mm | 400 mm |
| 12 mm | 600 mm | 480 mm |
| 16 mm | 800 mm | 640 mm |
| 20 mm | 1000 mm | 800 mm |
| 25 mm | 1250 mm | 1000 mm |

**Important Rules:**
1. **Stagger laps:** Not more than 50% bars spliced at one section
2. **Lap location:** Avoid at maximum moment locations
3. **Clear spacing:** At least 25mm or bar diameter (whichever greater)
4. **For bundled bars:** Calculate for equivalent diameter

**Additional Checks:**
- Add extra length at hooks/bends
- Increase by 25% for bundled bars
- Check drawing for specific requirements

**Reference:** IS 456:2000, Clause 26.2.5
"""
        return response
    
    def _handle_slump_query(self, question: str, context: dict) -> str:
        """Handle slump and workability queries."""
        slump_data = IS_CODES["IS 10262:2019"]["key_provisions"]["slump_values"]
        
        response = """**Answer for Site Engineer:**

## Slump Test & Workability Guide

**Recommended Slump Values (IS 456:2000, IS 10262:2019):**

| Application | Slump Range |
|-------------|-------------|
| Mass concrete | 25-50 mm |
| Normal RCC work | 50-100 mm |
| Pumped concrete | 75-100 mm |
| Trench fill / Piles | 100-150 mm |
| Congested reinforcement | 75-125 mm |

**Slump Test Procedure:**
1. Clean and dampen the cone
2. Fill in 3 equal layers
3. Rod each layer 25 times with 16mm rod
4. Strike off top, lift cone vertically (5-10 sec)
5. Measure slump immediately
6. Check slump type: True/Shear/Collapse

**Acceptance Criteria:**
- ± 25mm for specified slump ≤ 75mm
- ± 1/3 of specified value for slump > 75mm

**Common Issues:**
- Low slump → Add plasticizer (NOT water!)
- High slump → Check water content, reject if too high
- Shear slump → Indicates poor mix, re-mix and retest

**Quality Action:**
- Test at plant AND at site
- Record time of testing
- Reject concrete outside tolerance

**Reference:** IS 1199:1959, IS 456:2000
"""
        return response
    
    def _handle_cement_query(self, question: str, context: dict) -> str:
        """Handle cement-related queries."""
        cement_types = MATERIAL_SPECS["cement"]["types"]
        
        response = """**Answer for Site Engineer:**

## Cement Types & Selection Guide

**Common Cement Types:**

| Type | Best For | Key Property |
|------|----------|-------------|
| OPC 43 | General RCC | Standard strength |
| OPC 53 | High strength concrete | Early strength |
| PPC | Mass concrete, foundations | Lower heat, sulphate resistant |
| PSC | Marine works | High sulphate resistance |
| SRC | Aggressive soil exposure | Sulphate resisting |

**Selection Guidelines:**
- **Normal RCC work:** OPC 43 or OPC 53
- **Mass concrete (Raft/Mat):** PPC (lower heat)
- **Marine/Coastal:** PSC or SRC
- **Underground structures:** PPC or PSC
- **Precast elements:** OPC 53

**Storage Requirements:**
- Store on raised platform (150mm minimum)
- Stack maximum 10 bags high
- Keep away from moisture
- Use FIFO method
- Check manufacturing date (use within 3 months)

**Field Quality Checks:**
- No lumps when rubbed between fingers
- Greenish grey color (not white/yellow)
- Cool when put in palm
- Smooth texture
- Verify test certificate

**Reference:** IS 269, IS 455, IS 1489, IS 12269
"""
        return response
    
    def _handle_aggregate_query(self, question: str, context: dict) -> str:
        """Handle aggregate-related queries."""
        response = """**Answer for Site Engineer:**

## Aggregate Quality Guide

### Coarse Aggregate (Jelly/Gitti)

**Size Selection:**
| Application | Nominal Size |
|-------------|-------------|
| Normal RCC | 20 mm |
| Mass concrete | 40 mm |
| Congested sections | 10-12 mm |
| Thin sections | 10 mm or less |

**Quality Requirements:**
- Shape: Angular/Cubical (flakiness < 25%)
- Impact value: < 30% for wearing surfaces
- Crushing value: < 30% for concrete roads
- Water absorption: < 2%
- Clean, free from organic matter

### Fine Aggregate (Sand)

**Zone Classification (IS 383):**
- Zone I: Very coarse (rarely used alone)
- Zone II: Preferred for concrete
- Zone III: Acceptable
- Zone IV: Very fine (increase cement)

**Quality Requirements:**
- Silt content: < 3%
- Organic impurities: Nil
- FM: 2.6 - 2.9 preferred
- No clay lumps

**Field Tests:**

1. **Silt Content Test:**
   - Fill jar with sand to 50ml mark
   - Add water to 100ml, shake well
   - Let settle for 3 hours
   - Silt layer should be < 3% of sand volume

2. **Bulking Test:**
   - Measure dry sand volume
   - Add 4-8% water, measure again
   - Check bulking factor (max around 6% moisture)

**Reference:** IS 383:2016
"""
        return response
    
    def _handle_wcr_query(self, question: str, context: dict) -> str:
        """Handle water-cement ratio queries."""
        exposure = context.get("exposure_condition", "").lower() or "moderate"
        wcr_data = IS_CODES["IS 456:2000"]["key_provisions"]["max_water_cement_ratio"]
        
        response = f"""**Answer for Site Engineer:**

## Water-Cement Ratio (W/C Ratio) Guide

**Maximum W/C Ratio by Exposure (IS 456:2000, Table 5):**

| Exposure | Max W/C Ratio |
|----------|---------------|
| Mild | 0.55 |
| Moderate | 0.50 |
| Severe | 0.45 |
| Very Severe | 0.45 |
| Extreme | 0.40 |

**For {exposure.title()} Exposure:** Maximum **{wcr_data.get(exposure.replace(' ', '_'), 0.50)}**

**Why W/C Ratio Matters:**
- Lower W/C = Higher strength + Better durability
- Higher W/C = More porous, less durable
- Directly affects permeability and carbonation

**Practical Tips:**
1. Always use mix design approved values
2. Account for moisture in aggregates
3. Use plasticizer to reduce water (not add water for workability!)
4. Never add water at site without engineer approval

**Quality Control:**
- Record water added at plant
- Monitor slump to detect excess water
- Document any adjustments

**Reference:** IS 456:2000, Clause 8.2.4.2
"""
        return response
    
    def _handle_formwork_query(self, question: str, context: dict) -> str:
        """Handle formwork queries."""
        response = """**Answer for Site Engineer:**

## Formwork / Shuttering Guide

### Minimum Stripping Time (IS 456:2000, Table 11):

| Member | Min Days (OPC at 20°C) |
|--------|------------------------|
| Vertical (walls, columns) | 16-24 hours |
| Slab soffit (props remain) | 3 days |
| Beam soffit (props remain) | 7 days |
| Props to slabs (< 4.5m) | 7 days |
| Props to slabs (> 4.5m) | 14 days |
| Props to beams (< 6m) | 14 days |
| Props to beams (> 6m) | 21 days |

**For PPC/Lower temperatures:** Increase by 50%

### Pre-Pour Checklist:
- [ ] Formwork aligned and plumb
- [ ] Joints sealed (no grout leakage)
- [ ] Props at correct spacing
- [ ] Release agent applied
- [ ] Camber provided for beams > 6m
- [ ] Check dimensions against drawing

### Quality Points:
- Surface finish: Match specification
- Tolerances: ±3mm for exposed concrete
- Check deflection: Should not exceed span/300
- Water tightness: No grout loss

### Common Issues:
| Problem | Cause | Prevention |
|---------|-------|------------|
| Honeycombing at joints | Poor sealing | Use foam strips/sealant |
| Blow-out | Insufficient ties | Check form tie spacing |
| Bulging | Weak props | Follow prop design |

**Reference:** IS 456:2000, IS 14687:1999
"""
        return response
    
    def _handle_steel_query(self, question: str, context: dict) -> str:
        """Handle reinforcement steel queries."""
        response = """**Answer for Site Engineer:**

## Reinforcement Steel Guide

### Steel Grades (IS 1786:2008):

| Grade | Yield Strength | Common Use |
|-------|---------------|------------|
| Fe 415 | 415 N/mm² | General RCC |
| Fe 500 | 500 N/mm² | Standard use |
| Fe 500D | 500 N/mm² | Ductile, seismic zones |
| Fe 550 | 550 N/mm² | High strength needs |

### Quality Verification:
1. **Visual Check:**
   - Ribs clearly visible
   - Grade marking on bar (ISI mark, manufacturer, grade)
   - No heavy rust scaling

2. **Documentation:**
   - Mill test certificate
   - Weight per meter matches
   - Chemical composition in limits

3. **Site Test (if required):**
   - Bend test: Bend around mandrel (3d for Fe 500)
   - Should not crack at bend

### Minimum Bend Diameter:
| Bar Type | Bend Diameter |
|----------|---------------|
| Stirrups up to 20mm | 4 × bar dia |
| Main bars up to 20mm | 5 × bar dia |
| Main bars > 20mm | 6 × bar dia |

### Binding Wire:
- 18 gauge black annealed wire
- Approximately 8-10 kg per ton of steel

### Storage Requirements:
- Keep off ground (on timber supports)
- Under cover (avoid rain exposure)
- Separate by grade and diameter
- Label clearly

**Reference:** IS 1786:2008, IS 2502:1963
"""
        return response
    
    def _handle_testing_query(self, question: str, context: dict) -> str:
        """Handle testing-related queries."""
        response = """**Answer for Site Engineer:**

## Concrete Testing Guide

### Cube Testing (IS 516)

**Cube Preparation:**
- Mould size: 150mm × 150mm × 150mm
- Number: 3 cubes per sample (6 for 7 & 28 day)
- Casting: 3 layers, 35 blows each with tamping rod
- Label: Date, grade, location, mix ID

**Sampling Frequency (IS 456:2000):**
| Quantity | Minimum Samples |
|----------|-----------------|
| 1-5 m³ | 1 sample |
| 6-15 m³ | 2 samples |
| 16-30 m³ | 3 samples |
| 31-50 m³ | 4 samples |
| 50+ m³ | 4 + 1 per 50 m³ |

**Curing:**
- Demould after 24 hours
- Cure in water at 27 ± 2°C
- Mark specimens clearly

**Testing Schedule:**
- 7 days: Indicative (65-70% of 28-day strength)
- 28 days: Characteristic strength verification

### Acceptance Criteria (IS 456:2000, Table 11):

| fck (N/mm²) | Individual Min | Mean of 3 (Min) |
|-------------|----------------|-----------------|
| M20 | 15.0 | 20.0 + 0.825σ |
| M25 | 18.75 | 25.0 + 0.825σ |
| M30 | 22.5 | 30.0 + 0.825σ |

**If Results Fail:**
1. Identify affected area
2. Conduct core test if needed
3. Load test as last resort
4. Consult structural engineer

### Other Field Tests:
| Test | Purpose | Frequency |
|------|---------|-----------|
| Slump | Workability | Every truck |
| Temperature | Hot/Cold weather | Daily |
| Air content | For air-entrained | As specified |

**Reference:** IS 516:1959, IS 456:2000
"""
        return response
    
    def _handle_general_query(self, question: str, context: dict) -> str:
        """Handle general queries with comprehensive guidance."""
        response = f"""**Answer for Site Engineer:**

Thank you for your query. Based on your question about:
*"{question[:200]}..."*

Here's my guidance:

## General Quality Control Best Practices

### Before Starting Any Activity:
1. ✅ Check approved drawings are available
2. ✅ Verify material test certificates
3. ✅ Review method statement / ITP
4. ✅ Ensure tools and equipment ready
5. ✅ Brief the workforce

### During Execution:
1. ✅ Follow approved procedures
2. ✅ Monitor quality at each stage
3. ✅ Document with photos
4. ✅ Address issues immediately
5. ✅ Maintain checklists

### Key IS Codes for Reference:
- **IS 456:2000** - Concrete (most important)
- **IS 1786:2008** - TMT Steel
- **IS 10262:2019** - Mix Design
- **IS 2502:1963** - Bar Bending
- **IS 14687:1999** - Quality Assurance

### Need Specific Help?
Ask me about:
- Concrete grades and mix design
- Cover requirements
- Curing procedures
- Defect prevention
- Steel inspection
- Testing requirements
- Checklists

**Tip:** For specific answers, mention:
- Member type (slab, beam, column, footing)
- Concrete grade (M20, M25, M30...)
- Exposure condition (mild, moderate, severe...)
- Cement type (OPC, PPC, PSC)
"""
        return response
    
    def _format_checklist(self, checklist: dict) -> str:
        """Format a single checklist as markdown."""
        response = f"""**Answer for Site Engineer:**

## {checklist['title']}

"""
        for i, item in enumerate(checklist['items'], 1):
            response += f"- [ ] {item}\n"
        
        response += "\n**Action Required:** Complete all items before proceeding to next stage.\n"
        response += "**Documentation:** Sign and date checklist, attach to daily report.\n"
        return response
    
    def _format_all_checklists(self) -> str:
        """Format all available checklists."""
        response = """**Answer for Site Engineer:**

## Available Quality Checklists

I can provide detailed checklists for:

1. **Pre-Concreting** - Before pouring concrete
2. **During Concreting** - Active pouring checks  
3. **Post-Concreting** - Curing and finishing
4. **Steel Inspection** - Reinforcement verification
5. **Formwork** - Shuttering checks

**Ask specifically for any checklist, e.g.:**
- "Give me pre-concreting checklist"
- "What to check during concreting?"
- "Steel inspection points"

"""
        return response
    
    def _fallback_response(self, prompt: str) -> str:
        """Provide a helpful fallback response."""
        return """**Answer for Site Engineer:**

I understand you have a question about civil quality control. Let me help you with the most common topics:

## How I Can Help:

1. **Concrete Quality**
   - Grade selection by exposure
   - Curing requirements
   - Cover specifications
   - Mix design queries

2. **Inspections & Checklists**
   - Pre/During/Post concreting
   - Steel inspection
   - Formwork checks

3. **Defect Prevention**
   - Honeycomb, cold joints
   - Cracks, segregation
   - Remediation steps

4. **Testing Requirements**
   - Cube testing
   - Slump test
   - Acceptance criteria

5. **Material Specifications**
   - Cement types
   - Aggregate quality
   - Steel grades

**Please rephrase your question with specific details like:**
- What member? (slab, beam, column, footing)
- What grade? (M20, M25, M30...)
- What exposure? (mild, moderate, severe...)

I'm here to provide practical, IS code-based guidance!
"""
    
    def build_prompt(self, project_info: Dict[str, Any], web_context: List[Dict[str, str]], user_question: str) -> str:
        """
        Construct the runtime prompt.
        """
        # Format Web Context
        context_str = ""
        for item in web_context:
            context_str += f"- Title: {item['title']}\n  Snippet: {item['snippet']}\n  URL: {item['url']}\n\n"
        
        if not context_str:
            context_str = "No external search results found."

        # Format Project Info
        info_str = json.dumps(project_info, indent=2)

        return f"""
PROJECT_INFO:
{info_str}

WEB_CONTEXT:
{context_str}

USER_QUESTION:
{user_question}

Now:
- First internally analyze relevance and consistency
- Then respond in this format:

Answer for Site Engineer:
...

Reasoning (short bullets):
- ...
- ...

Sources Used:
- URL1
- URL2
"""


# Singleton instance - now uses the self-contained expert
llm_client = CivilQualityExpert()

