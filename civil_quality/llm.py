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
    "IS 1905:1987": {
        "title": "Code of Practice for Structural Use of Unreinforced Masonry",
        "topics": ["masonry", "brickwork", "blockwork", "mortar"],
        "key_provisions": {
            "mortars": "Mix proportions and workmanship for mortar and brickwork",
            "tolerances": "Verticality and alignment tolerances",
            "reinforcement": "Where reinforced masonry is required follow structural design"
        }
    },
    "IS 1077": {
        "title": "Common Burnt Clay Building Bricks - Specification",
        "topics": ["brick", "masonry", "brick quality"],
        "key_provisions": {
            "size": "Standard brick dimensions and tolerance",
            "strength": "Minimum compressive strength grades",
            "water_absorption": "Maximum permissible" 
        }
    },
    "IS 3043:1987": {
        "title": "Code of Practice for Earthing",
        "topics": ["earthing", "grounding", "electrical"],
        "key_provisions": {
            "earth_resistance": "Max earth pit resistance per type of installation",
            "electrode_types": "Use copper rods/plates depending on soil",
            "bonding": "Equipotential bonding and earthing arrangement"
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
    ,
    "IS 383:2016": {
        "title": "Specification for Coarse and Fine Aggregates from Natural Sources for Concrete",
        "topics": ["aggregate", "coarse", "fine", "sieve analysis"],
        "key_provisions": {
            "grading": "As per zones; limits on silt and flakiness",
            "max_size": "20mm typical for RCC",
            "tests": ["sieve analysis", "silt content", "crushing value", "impact value"]
        }
    },
    "IS 516:1959": {
        "title": "Methods of Tests for Strength of Concrete",
        "topics": ["testing", "cube test", "compressive strength"],
        "key_provisions": {
            "cube_size": "150 x 150 x 150 mm standard",
            "curing": "Specimens cured at 27 +/- 2 deg C",
            "testing_age": "7-day and 28-day tests standard"
        }
    },
    "IS 269:2015": {
        "title": "Specification for Ordinary Portland Cement, 33 grade (OPC 33)",
        "topics": ["cement", "opc 33", "cement quality"],
        "key_provisions": {
            "fineness": "as per test",
            "soundness": "LeChatelier method",
            "setting_time": "Initial & final limits"
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


# ==================== TOPIC GUIDES ====================
TOPIC_GUIDES = {
    "plastering": {
        "title": "Plastering & Finishes",
        "summary": "Includes gypsum plaster, cement plaster, POP, putty and finishing",
        "key_points": [
            "Prepare substrate: clean, remove loose material",
            "Use bonding agent where required",
            "Control thickness: 10-15mm single coat for gypsum; cement plaster 12-20mm",
            "Mix water-cement ratio per manufacturer's guidelines",
            "Allow adequate drying and curing time as per product"
        ],
        "checklists": ["formwork", "pre_concreting", "post_concreting"],
        "related_is_codes": ["IS 456:2000", "IS 383:2016"]
    },
    "painting": {
        "title": "Painting & Coatings",
        "summary": "Surface preparation, primer, paint application, and curing/drying",
        "key_points": [
            "Ensure substrate is clean and dry",
            "Apply primer where specified",
            "Follow recommended number of coats and drying times",
            "Beware of ambient temperature/humidity during application"
        ],
        "checklists": [],
        "related_is_codes": []
    },
    "tiling": {
        "title": "Tiling & Stone Works",
        "summary": "Selection of tiles, surface preparation, adhesives, and grouting",
        "key_points": [
            "Surface levelness: max 3mm variation in 2m for floors",
            "Use appropriate adhesive and back-buttering for large tiles",
            "Allow adequate curing before grouting",
            "Check tile dimensions and shade prior to fixing"
        ],
        "checklists": [],
        "related_is_codes": []
    },
    "waterproofing": {
        "title": "Waterproofing",
        "summary": "Membrane types, cementitious and integral waterproofing and testing",
        "key_points": [
            "Repair and prime substrate before application",
            "Apply required number of coats and lap details",
            "Perform water test (48-72 hours) post application",
            "Protect membrane during tiling or screeding"
        ],
        "checklists": [],
        "related_is_codes": ["IS 456:2000"]
    },
    "masonry": {
        "title": "Masonry & Brickwork",
        "summary": "Brick/block selection, mortar, joints, reinforcement and tolerances",
        "key_points": [
            "Use appropriate mortar mix as per design",
            "Check brick quality and cleanliness",
            "Ensure proper curing and joint finishing",
            "Follow reinforcement details for reinforced masonry"
        ],
        "checklists": [],
        "related_is_codes": ["IS 1905:1987", "IS 1077"]
    },
    "plumbing": {
        "title": "Plumbing & Drainage",
        "summary": "Pipe selection, jointing, slope, testing and trap/sewage considerations",
        "key_points": [
            "Select pipes and materials as per application (SWR, CPVC, uPVC, PPR)",
            "Maintain correct slope in drainage runs",
            "Perform hydro testing for potable/pressure lines",
            "Inspect for proper jointing and embedment protection"
        ],
        "checklists": [],
        "related_is_codes": []
    },
    "electrical": {
        "title": "Electrical Works",
        "summary": "Conductors, conduits, earthing, switches, and installations",
        "key_points": [
            "Ensure proper sizing of conductors and protective devices",
            "Provide adequate earthing as per IS 3043",
            "Use certified components and test insulation",
            "Follow cable routing, protection and labeling norms"
        ],
        "checklists": [],
        "related_is_codes": ["IS 3043:1987"]
    },
    "doors_windows": {
        "title": "Doors & Windows",
        "summary": "Frames, shutters, tolerances, fixings, and finishes",
        "key_points": [
            "Check sizes against drawings",
            "Ensure proper alignment and plumb",
            "Use proper fixings and anchors per frame type",
            "Test functioning: swing, lock and hardware"
        ],
        "checklists": [],
        "related_is_codes": []
    },
    "safety": {
        "title": "Site Safety & PPE",
        "summary": "Scaffold safety, fall protection, PPE, and emergency procedures",
        "key_points": [
            "Use scaffold per manufacturer and safety codes",
            "Provide and enforce PPE usage",
            "Keep signage and emergency contact numbers visible",
            "Regular safety inductions for workers"
        ],
        "checklists": [],
        "related_is_codes": []
    },
    "earthwork": {
        "title": "Earthwork & Compaction",
        "summary": "Excavation, stratification, compaction control and fill materials",
        "key_points": [
            "Check initial soil classification and water table",
            "Follow compaction specifications and testing (Proctor/CBR)",
            "Provide erosion control and dewatering where necessary",
            "Inspect bearing strata prior to foundation"],
        "checklists": [],
        "related_is_codes": []
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
        # include topic guides for broader knowledge coverage
        self.knowledge_base["topic_guides"] = TOPIC_GUIDES

    def get_kb_index(self) -> Dict[str, Any]:
        """Return an index of the available knowledge base items (IS codes and topics)."""
        return {
            "is_codes": list(self.knowledge_base.get("is_codes", {}).keys()),
            "topics": list(self.knowledge_base.get("topic_guides", {}).keys()),
            "checklists": list(self.knowledge_base.get("checklists", {}).keys())
        }
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response using rule-based logic and knowledge base.
        No external API required.
        """
        try:
            # Parse the user prompt to extract context, web_context and question
            context, web_context, question = self._parse_prompt(user_prompt)
            
            # Attach web_context to context (for fallback logic)
            context['web_context'] = web_context

            # Generate intelligent response
            response = self._generate_response(question, context)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(user_prompt)
    
    def _parse_prompt(self, prompt: str) -> tuple:
        """Parse the structured prompt to extract context and question."""
        context = {}
        web_context: List[Dict[str, str]] = []
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
        
        # Extract WEB_CONTEXT section - look for our formatted block
        if "WEB_CONTEXT:" in prompt:
            try:
                start_w = prompt.find("WEB_CONTEXT:") + len("WEB_CONTEXT:")
                end_w = prompt.find("USER_QUESTION:", start_w)
                raw_web = prompt[start_w:end_w].strip()
                # Parse simple lines we created in build_prompt
                # Each block begins with '- Title: ' and contains Snippet and URL
                lines = [l.strip() for l in raw_web.splitlines() if l.strip()]
                cur = {}
                for l in lines:
                    if l.startswith("- Title:"):
                        if cur:
                            web_context.append(cur)
                        cur = {"title": l.replace("- Title:", "").strip(), "snippet": "", "url": ""}
                    elif l.startswith("Snippet:"):
                        cur["snippet"] = l.replace("Snippet:", "").strip()
                    elif l.startswith("URL:"):
                        cur["url"] = l.replace("URL:", "").strip()
                if cur:
                    web_context.append(cur)
            except Exception:
                web_context = []

        # Extract USER_QUESTION section
        if "USER_QUESTION:" in prompt:
            start = prompt.find("USER_QUESTION:") + len("USER_QUESTION:")
            end = prompt.find("Now:", start)
            if end == -1:
                end = len(prompt)
            question = prompt[start:end].strip()
        else:
            question = prompt
        
        return context, web_context, question

    def _is_question_in_kb(self, question: str) -> Dict[str, List[str]]:
        """Check knowledge base for direct coverage on the question.
        Returns mapping of KB areas and keys that matched the question.
        """
        q_lower = question.lower()
        matches = {"is_codes": [], "issues": [], "checklists": [], "materials": [], "topics": []}

        # Check IS codes topics and keys
        for code, spec in self.knowledge_base.get("is_codes", {}).items():
            # Match code name or topics
            if code.lower() in q_lower:
                matches["is_codes"].append(code)
            else:
                for t in spec.get("topics", []):
                    if t.lower() in q_lower:
                        matches["is_codes"].append(code)
                        break

        # Check quality issues
        for issue in self.knowledge_base.get("quality_issues", {}).keys():
            if issue in q_lower:
                matches["issues"].append(issue)

        # Check checklists
        for ck in self.knowledge_base.get("checklists", {}).keys():
            if ck.replace("_", " ") in q_lower or ck in q_lower:
                matches["checklists"].append(ck)

        # Check materials
        for mat in self.knowledge_base.get("materials", {}).keys():
            if mat in q_lower:
                matches["materials"].append(mat)

        # Check topic guides (plastering, tiling, painting, etc.)
        for t_key, t_val in self.knowledge_base.get("topic_guides", {}).items():
            # check title and keywords
            if t_key in q_lower or any(w in q_lower for w in t_val.get("title", "").lower().split()):
                matches["topics"].append(t_key)

        return matches

    def _handle_web_fallback(self, question: str, context: dict, web_context: List[Dict[str, str]]) -> str:
        """Summarize web search results as fallback when question is outside our KB."""
        # Simple summarization: collect top snippets and URLs
        parts = ["**Answer (from web sources)**\n\n"]
        parts.append(f"I couldn't find a direct match in the internal IS Code knowledge base for: **{question}**.\n")
        parts.append("Here are summarized web sources that may help (please verify against relevant IS Codes for compliance):\n")

        for w in web_context[:3]:
            title = w.get("title", "(no title)")
            snippet = w.get("snippet", "")
            url = w.get("url", "")
            parts.append(f"- {title}: {snippet} ({url})\n")

        # Highlight any IS codes found in the web result snippets/titles
        found_is_codes = set()
        for w in web_context:
            txt = (w.get('title', '') + ' ' + w.get('snippet', '')).upper()
            for code in self.knowledge_base.get('is_codes', {}).keys():
                if code.upper() in txt:
                    found_is_codes.add(code)

        if found_is_codes:
            parts.append("\nThe following IS codes were referenced in the web results:\n")
            for c in sorted(found_is_codes):
                parts.append(f"- {c}\n")

        parts.append("\nRecommendation: Cross-check the above sources against applicable IS Codes (e.g., IS 456, IS 10262) for final verification.\n")

        return "\n".join(parts)
    
    def _generate_response(self, question: str, context: dict) -> str:
        """Generate intelligent response based on question and context."""
        q_lower = question.lower()
        # Check if the KB covers this topic; if not, web fallback will be used
        kb_matches = self._is_question_in_kb(question)

        # If the question mentions emerging technologies/materials, prefer web search for latest info
        emerging_keywords = ["graphene", "nanotech", "nanotechnology", "nanocoating", "epoxy", "composite", "polymer", "coating", "coated"]
        if any(k in question.lower() for k in emerging_keywords) and context.get('web_context'):
            return self._handle_web_fallback(question, context, context.get('web_context'))
        
        # If the KB has matches for this topic but we didn't go into a specific handler
        # we'll generate a KB-based response.
        if any(kb_matches.values()):
            # If a specific handler triggers below, that will be used, otherwise fallback to KB summary
            pass

        # QUICK COMMANDS
        if any(kw in q_lower for kw in ["list is codes", "list is code", "knowledge base", "list kb", "show kb"]):
            return self._handle_kb_index()

        # ===== FINISHING WORKS =====
        if any(word in q_lower for word in ["gypsum", "plaster", "plastering", "pop", "putty"]):
            return self._handle_plastering_query(question, context)
        
        elif any(word in q_lower for word in ["paint", "painting", "primer", "emulsion", "enamel", "distemper"]):
            return self._handle_painting_query(question, context)
        
        elif any(word in q_lower for word in ["tile", "tiling", "ceramic", "vitrified", "marble", "granite", "flooring", "dado"]):
            return self._handle_tiling_query(question, context)
        
        elif any(word in q_lower for word in ["waterproof", "water proof", "waterproofing", "leakage", "seepage", "damp"]):
            return self._handle_waterproofing_query(question, context)
        
        # ===== MASONRY & BLOCKWORK =====
        elif any(word in q_lower for word in ["brick", "block", "masonry", "aac", "fly ash", "mortar"]):
            return self._handle_masonry_query(question, context)
        
        # ===== MEP WORKS =====
        elif any(word in q_lower for word in ["plumbing", "pipe", "cpvc", "upvc", "sanitary", "drainage", "sewer", "fitting"]):
            return self._handle_plumbing_query(question, context)
        
        elif any(word in q_lower for word in ["electrical", "wiring", "conduit", "cable", "switch", "socket", "earthing", "grounding"]):
            return self._handle_electrical_query(question, context)
        
        # ===== DOORS & WINDOWS =====
        elif any(word in q_lower for word in ["door", "window", "frame", "shutter", "upvc window", "aluminium", "aluminum"]):
            return self._handle_doors_windows_query(question, context)
        
        # ===== CONCRETE RELATED =====
        elif any(word in q_lower for word in ["checklist", "check list"]):
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
        
        elif any(word in q_lower for word in ["rebar", "reinforcement", "steel", "bar", "tmt"]):
            return self._handle_steel_query(question, context)
        
        elif any(word in q_lower for word in ["test", "cube", "cylinder", "sample"]):
            return self._handle_testing_query(question, context)
        
        elif any(word in q_lower for word in ["concrete", "concreting", "rcc", "pcc"]):
            return self._handle_concrete_query(question, context)
        
        # ===== OTHER WORKS =====
        elif any(word in q_lower for word in ["scaffold", "scaffolding", "safety", "ppe", "fall"]):
            return self._handle_safety_query(question, context)
        
        elif any(word in q_lower for word in ["excavation", "earthwork", "backfill", "compaction", "soil"]):
            return self._handle_earthwork_query(question, context)
        
        else:
            # Not covered directly by internal KB handlers
            # But if KB has matches, return a KB-based summary
            if any([len(v) for v in kb_matches.values()]):
                return self._handle_kb_query(question, context, kb_matches)
            # 1. If the KB has matches, provide KB-based answer
            if any(kb_matches.values()):
                return self._handle_kb_query(question, context, kb_matches)

            # 2. If web context (search results) are present, summarize them
            if 'web_context' in context and context.get('web_context'):
                return self._handle_web_fallback(question, context, context.get('web_context'))

            # 3. Fall back to general guidance
            return self._handle_general_query(question, context)
    
    # ==================== PLASTERING & FINISHING ====================
    
    def _handle_plastering_query(self, question: str, context: dict) -> str:
        """Handle gypsum plaster, cement plaster, POP, putty queries."""
        q_lower = question.lower()
        
        if "gypsum" in q_lower:
            return """**Answer for Site Engineer:**

## Gypsum Plaster - Complete Application Guide

### What is Gypsum Plaster?
Gypsum plaster is a white binding material made from gypite mineral. It provides smooth finish, faster drying, and no curing required.

### Advantages over Cement Plaster:
- ✅ No curing required (saves water and time)
- ✅ Faster application and setting
- ✅ Smooth finish - ready for painting
- ✅ No shrinkage cracks
- ✅ Better thermal insulation
- ✅ Fire resistant

### Step-by-Step Application Process:

**1. Surface Preparation:**
- Clean the surface thoroughly (remove dust, loose particles)
- Hack the concrete surface for better bonding
- Apply bonding agent/primer coat if needed
- Ensure surface is dry (no dampness)
- Fix chicken mesh at RCC-brick junctions

**2. Mixing:**
- Use clean potable water
- Add gypsum powder to water (not water to powder)
- Mix ratio: approximately 50-60% water by weight
- Mix for 2-3 minutes until lump-free
- Use within 20-25 minutes (pot life)

**3. Application:**
- Apply in single coat (12-15mm thickness)
- For walls: bottom to top
- Use straight edge/aluminum rule to level
- For thickness > 15mm, apply in two coats
- Second coat after first coat sets (not fully dry)

**4. Finishing:**
- After 10-15 minutes of application, do sponge finishing
- Use trowel for smooth surface
- For textured finish, use appropriate tools

### Thickness Guidelines:
| Surface | Thickness |
|---------|-----------|
| RCC Ceiling | 6-8 mm |
| Brick Wall | 11-15 mm |
| RCC Wall | 8-12 mm |
| Columns | 8-12 mm |

### Quality Checkpoints:
- [ ] Surface must be dry (no moisture)
- [ ] No dust or loose particles
- [ ] Chicken mesh at joints
- [ ] Correct water-powder ratio
- [ ] Single coat thickness ≤ 15mm
- [ ] Level and plumb checked
- [ ] No hollowness (tap test)
- [ ] Corners and edges sharp

### Common Defects & Prevention:
| Defect | Cause | Prevention |
|--------|-------|------------|
| Hollow patches | Poor bonding | Clean surface, use primer |
| Cracks | Thick coat | Limit to 15mm, use mesh |
| Debonding | Wet surface | Ensure dry substrate |
| Uneven finish | Poor workmanship | Use skilled plasterer |

### DO's and DON'Ts:

**DO's:**
- ✅ Apply on dry surfaces only
- ✅ Use within pot life (20-25 min)
- ✅ Mix small batches
- ✅ Clean tools immediately
- ✅ Store bags in dry place

**DON'Ts:**
- ❌ Don't apply on wet surfaces
- ❌ Don't add more water after mixing
- ❌ Don't reuse leftover material
- ❌ Don't apply in single coat > 15mm
- ❌ Don't use external (use only internal)

### Drying Time:
- Touch dry: 25-30 minutes
- Ready for painting: 3-4 days
- Full strength: 72 hours

### Reference Standards:
- IS 2547 (Part 1): Gypsum building plasters
- IS 2547 (Part 2): Gypsum plaster boards
"""
        
        elif "putty" in q_lower:
            return """**Answer for Site Engineer:**

## Wall Putty - Application Guide

### Types of Putty:
1. **White Cement Putty** - Water resistant, for exteriors
2. **Acrylic Putty** - Better binding, for interiors

### Application Process:

**1. Surface Preparation:**
- Plaster must be cured (min 14 days for cement plaster)
- Surface should be dry and clean
- Remove loose particles with sandpaper
- Apply primer if surface is porous

**2. First Coat:**
- Mix putty with 30-40% water
- Apply thin coat (1-1.5mm)
- Allow 4-6 hours drying
- Sand with 180 grit sandpaper

**3. Second Coat:**
- Apply after first coat is dry
- Thickness: 0.5-1mm
- Total thickness: 2-3mm max
- Sand with 320-400 grit paper

### Coverage:
- First coat: 20-25 sq.ft per kg
- Second coat: 30-35 sq.ft per kg

### Quality Checks:
- [ ] No undulations on surface
- [ ] Smooth to touch
- [ ] No visible brush/trowel marks
- [ ] Corners sharp and straight

**Reference:** IS 15477: White Cement Based Putty
"""
        
        else:  # General cement plaster
            return """**Answer for Site Engineer:**

## Cement Plaster - Complete Guide

### Mix Ratios:
| Location | Mix Ratio | Thickness |
|----------|-----------|-----------|
| External walls | 1:4 | 15-20mm |
| Internal walls | 1:5 or 1:6 | 12-15mm |
| Ceiling | 1:4 | 6-8mm |
| Dado | 1:3 | 12mm |

### Application Process:

**1. Surface Preparation:**
- Clean surface of dust, oil, loose particles
- Rake joints to 10-12mm depth
- Wet the surface thoroughly
- Fix dot and strip (screeds) for level

**2. Application (Two Coat Work):**

*First Coat (Rendering):*
- Thickness: 10-12mm
- Apply in patches, level with wooden float
- Rough surface for second coat bonding
- Cure for 7 days

*Second Coat (Finishing):*
- Thickness: 3-5mm
- Apply after first coat is cured
- Finish with steel trowel
- Cure for 7 days

**3. Curing:**
- Start after initial set (24 hours)
- Continue for minimum 7 days
- Keep wet continuously

### Quality Checkpoints:
- [ ] Verticality checked with plumb bob
- [ ] Surface level within 3mm in 3m
- [ ] No hollow areas (tap test)
- [ ] Corners and edges sharp
- [ ] Even texture throughout

**Reference:** IS 1661, IS 2402
"""

    def _handle_painting_query(self, question: str, context: dict) -> str:
        """Handle painting related queries."""
        return """**Answer for Site Engineer:**

## Painting Work - Complete Guide

### Types of Paint & Uses:
| Type | Use | Coats |
|------|-----|-------|
| Primer | Base coat, sealing | 1 |
| Distemper | Economy internal | 2-3 |
| Emulsion | Interior walls | 2-3 |
| Exterior Emulsion | External walls | 2-3 |
| Enamel | Wood, metal | 2-3 |
| Cement Paint | Exteriors | 2 |

### Step-by-Step Process:

**1. Surface Preparation:**
- Surface must be dry (min 4 weeks after plastering)
- Putty applied and sanded smooth
- Remove dust with damp cloth
- Fill cracks with crack filler

**2. Primer Application:**
- Apply one coat of primer
- Coverage: 100-120 sq.ft/ltr
- Drying time: 4-6 hours
- Sand lightly after drying

**3. Paint Application:**
- First coat: slightly thinned (10% water)
- Drying time between coats: 4-6 hours
- Final coat: as per manufacturer
- Apply in one direction

### Coverage (per liter):
| Type | First Coat | Final Coat |
|------|-----------|------------|
| Interior Emulsion | 100-120 sq.ft | 120-140 sq.ft |
| Exterior Emulsion | 45-55 sq.ft | 60-80 sq.ft |
| Enamel | 100-120 sq.ft | 120-140 sq.ft |

### Quality Checkpoints:
- [ ] No visible brush marks
- [ ] Uniform color and sheen
- [ ] No drips or runs
- [ ] Edges clean and sharp
- [ ] No patches or misses
- [ ] Film thickness adequate

### Common Defects:
| Defect | Cause | Prevention |
|--------|-------|------------|
| Peeling | Moisture, poor adhesion | Ensure dry surface, use primer |
| Chalking | UV degradation | Use quality paint |
| Blistering | Moisture/heat | Proper surface prep |
| Flaking | Over old paint | Scrape and prime |
| Sagging | Thick application | Apply thin coats |

### DO's and DON'Ts:

**DO's:**
- ✅ Wait for surface to dry completely
- ✅ Apply in dry weather
- ✅ Stir paint thoroughly before use
- ✅ Maintain wet edge while painting
- ✅ Cover adjacent surfaces

**DON'Ts:**
- ❌ Don't paint in rain or high humidity
- ❌ Don't apply thick coats
- ❌ Don't mix different brands
- ❌ Don't skip primer

**Reference:** IS 428 (Distemper), IS 2395 (Emulsion)
"""

    def _handle_tiling_query(self, question: str, context: dict) -> str:
        """Handle tiling related queries."""
        return """**Answer for Site Engineer:**

## Tiling Work - Complete Guide

### Types of Tiles:
| Type | Use | Thickness |
|------|-----|-----------|
| Ceramic | Walls, light traffic | 6-8mm |
| Vitrified | Floors, high traffic | 8-12mm |
| Porcelain | Premium floors | 8-10mm |
| Natural Stone | Floors, cladding | 15-20mm |

### Step-by-Step Process:

**1. Surface Preparation:**
- Surface must be level (max 3mm variation in 2m)
- Clean and free from dust/oil
- Hack smooth concrete surfaces
- Wet the surface before tiling

**2. Setting Out:**
- Mark center lines
- Dry lay tiles to check pattern
- Plan cuts to be at edges/corners
- Maintain uniform joint width

**3. Tile Fixing:**

*For Wall Tiles:*
- Use tile adhesive (recommended) or cement slurry
- Apply adhesive with notched trowel
- Press tile firmly with slight twist
- Use spacers for uniform joints
- Start from bottom, work upward

*For Floor Tiles:*
- Spread adhesive/mortar on floor
- Back-butter large format tiles
- Use rubber mallet to level
- Check level frequently
- Allow 24 hours before grouting

**4. Grouting:**
- Clean joints of excess adhesive
- Apply grout with rubber float
- Work diagonally across joints
- Clean excess within 15-20 minutes
- Cure grout for 24 hours

### Joint Width Guidelines:
| Tile Size | Joint Width |
|-----------|-------------|
| Up to 300mm | 2-3mm |
| 300-600mm | 3-5mm |
| Above 600mm | 5-8mm |

### Quality Checkpoints:
- [ ] Level and plumb checked
- [ ] No lippage between tiles
- [ ] Uniform joint width
- [ ] No hollow tiles (tap test)
- [ ] Clean grout lines
- [ ] Pattern matching correct
- [ ] Cuts clean and at edges

### Coverage (Adhesive):
- Wall tiles: 4-5 kg/sq.m
- Floor tiles: 5-6 kg/sq.m
- Large format: 6-8 kg/sq.m

### Common Defects:
| Defect | Cause | Prevention |
|--------|-------|------------|
| Hollow tiles | Insufficient adhesive | Back-butter, full coverage |
| Lippage | Uneven substrate | Level surface, use leveling system |
| Cracked tiles | Point load, hollow | Avoid hollow, use flexible adhesive |
| Grout cracking | Wrong grout, movement | Use flexible grout at joints |

**Reference:** IS 13753 (Ceramic tiles), IS 15622 (Tile adhesive)
"""

    def _handle_waterproofing_query(self, question: str, context: dict) -> str:
        """Handle waterproofing related queries."""
        return """**Answer for Site Engineer:**

## Waterproofing - Complete Guide

### Types of Waterproofing:

| Type | Application | Use |
|------|-------------|-----|
| Integral | Mixed in concrete | Basements, tanks |
| Membrane | Applied on surface | Roofs, terraces |
| Cementitious | Brush/spray applied | Toilets, balconies |
| Crystalline | Penetrating | Water tanks, tunnels |
| Bituminous | Hot/cold applied | Below grade |

### Application by Area:

**1. Toilet/Bathroom Waterproofing:**

*Step-by-Step Process:*
1. Clean surface of debris, dust
2. Fill cracks with polymer morite
3. Apply primer coat (wait to dry)
4. Apply first coat of waterproofing membrane
5. Embed waterproofing tape at corners/joints
6. Apply second coat perpendicular to first
7. Apply screed/protection layer
8. Water test for 48 hours
9. Proceed with tiling

*Coverage:*
- Membrane: 1.2-1.5 sq.m per liter
- Total thickness: 1.5-2mm

**2. Terrace/Roof Waterproofing:**

*Steps:*
1. Clean surface, repair cracks
2. Apply primer
3. Apply waterproofing (2-3 coats)
4. Apply screed with slope (1:100)
5. Apply reflective/protective coat
6. Water test for 72 hours

**3. Basement Waterproofing:**
- Use integral waterproofing in concrete
- Apply crystalline coating
- Provide drain and sump pump
- External membrane if accessible

### Quality Checkpoints:
- [ ] Surface clean and dry
- [ ] Cracks treated before application
- [ ] Corners and joints reinforced with tape
- [ ] Specified thickness achieved
- [ ] Continuity maintained (no gaps)
- [ ] Water ponding test passed
- [ ] Protection layer applied before tiling

### Water Test Duration:
| Area | Duration |
|------|----------|
| Toilets | 48 hours |
| Terraces | 72 hours |
| Water tanks | 7 days |

### Common Defects:
| Defect | Cause | Prevention |
|--------|-------|------------|
| Leakage at joints | No tape reinforcement | Embed fabric/tape |
| Membrane damage | Traffic before protection | Apply screed immediately |
| Bubbling | Moisture in substrate | Ensure dry surface |
| Peeling | Poor adhesion | Clean surface, use primer |

**Reference:** IS 3036 (Bitumen sheets), IS 3384 (Integral compounds)
"""

    def _handle_masonry_query(self, question: str, context: dict) -> str:
        """Handle masonry and blockwork queries."""
        return """**Answer for Site Engineer:**

## Masonry & Blockwork - Complete Guide

### Types of Masonry Units:

| Type | Size (mm) | Density | Use |
|------|-----------|---------|-----|
| Red Brick | 230x110x75 | 1800 kg/m³ | Load bearing walls |
| Fly Ash Brick | 230x110x75 | 1700 kg/m³ | General masonry |
| AAC Block | 600x200x various | 550-650 kg/m³ | Non-load bearing |
| Solid Concrete Block | 400x200x200 | 2000 kg/m³ | Load bearing |
| Hollow Block | 400x200x200 | 1400 kg/m³ | Partition walls |

### Mortar Mix Ratios:
| Purpose | Cement:Sand |
|---------|-------------|
| Load bearing | 1:4 |
| Non-load bearing | 1:5 or 1:6 |
| Pointing | 1:3 |
| AAC Block | Use AAC adhesive |

### Step-by-Step Process:

**1. Setting Out:**
- Mark wall lines from drawing
- Check room dimensions
- Set corner blocks first
- String line for alignment

**2. Laying Bricks/Blocks:**
- Wet bricks before use (not AAC)
- Apply mortar bed (10-12mm)
- Place brick, tap to level
- Check plumb every course
- Stagger joints (running bond)
- Max 1.5m height per day

**3. Joints & Finish:**
- Rake joints for plastering
- Point joints for exposed brick
- Cure for 7 days

### Quality Checkpoints:
- [ ] Bricks/blocks soaked (except AAC)
- [ ] First course on PCC/RCC level
- [ ] Verticality checked every 3 courses
- [ ] Horizontal level maintained
- [ ] Joints staggered properly
- [ ] Wall thickness as per drawing
- [ ] Openings at correct location
- [ ] Tooth left for future walls

### Brick/Block Requirements:
- Bricks per sq.m (230x110x75): ~55 nos (with 10mm joint)
- AAC blocks per sq.m (600x200x200): ~8.3 nos

### Bond Types:
- **Stretcher Bond**: Most common, single wythe
- **English Bond**: Alternate header/stretcher courses
- **Flemish Bond**: Decorative, header+stretcher in same course

### Common Defects:
| Defect | Cause | Prevention |
|--------|-------|------------|
| Cracks at junctions | No chicken mesh | Embed mesh at RCC-brick joints |
| Bulging | Thick mortar, no curing | Control thickness, wet bricks |
| Efflorescence | Salt in bricks | Use quality bricks, good drainage |

**Reference:** IS 2212 (Brick Masonry), IS 2185 (Concrete Blocks)
"""

    def _handle_plumbing_query(self, question: str, context: dict) -> str:
        """Handle plumbing related queries."""
        return """**Answer for Site Engineer:**

## Plumbing Work - Complete Guide

### Pipe Types & Uses:

| Pipe Type | Use | Jointing |
|-----------|-----|----------|
| CPVC | Hot & cold water | Solvent cement |
| uPVC | Cold water, drainage | Solvent cement |
| PPR | Hot & cold water | Heat fusion |
| GI | External, fire line | Threading |
| HDPE | Underground | Heat fusion |
| SWR | Soil & waste | Ring seal |

### Pipe Sizing Guidelines:
| Application | Size |
|-------------|------|
| Wash basin | 32mm |
| Shower/floor drain | 75-100mm |
| WC outlet | 100mm |
| Kitchen sink | 40-50mm |
| Soil stack | 100-110mm |
| Water supply main | 25-32mm |

### Installation Process:

**1. Concealed Piping:**
- Chase walls after masonry curing
- Depth: pipe dia + 25mm cover
- Support pipes at intervals
- Test before closing
- Fill with weak mortar

**2. Exposed/External:**
- Use proper clamps/supports
- Allow for expansion
- Insulate hot water lines
- Provide access panels

**3. Testing:**
- Pressure test: 1.5x working pressure
- Duration: 30 minutes minimum
- Check all joints for leaks

### Quality Checkpoints:
- [ ] Pipes as per approved make/brand
- [ ] Proper slope for drainage (1:100)
- [ ] Supports at specified intervals
- [ ] Pressure test passed
- [ ] No cross-connection
- [ ] Venting provided
- [ ] Clean-outs accessible
- [ ] Sleeves at wall/slab crossings

### Slope Requirements:
| Pipe Size | Minimum Slope |
|-----------|---------------|
| 75mm | 1:50 (2%) |
| 100mm | 1:60 (1.67%) |
| 150mm | 1:100 (1%) |

### Support Spacing:
| Pipe Size | Horizontal | Vertical |
|-----------|-----------|----------|
| 20-25mm | 1.0m | 1.2m |
| 32-50mm | 1.2m | 1.5m |
| 75-110mm | 1.5m | 1.8m |

### Common Defects:
| Defect | Cause | Prevention |
|--------|-------|------------|
| Leakage at joints | Poor jointing | Proper solvent/fusion |
| Blockage | Insufficient slope | Maintain 1:100 minimum |
| Water hammer | No air cushion | Install arrestors |
| Cross contamination | Wrong connections | Color code pipes |

**Reference:** IS 4985 (uPVC), IS 15778 (CPVC), IS 15801 (PPR)
"""

    def _handle_electrical_query(self, question: str, context: dict) -> str:
        """Handle electrical work queries."""
        return """**Answer for Site Engineer:**

## Electrical Work - Complete Guide

### Cable Sizing (Common):
| Load | Cable Size | MCB |
|------|-----------|-----|
| Lighting (1 circuit) | 1.5 sq.mm | 6A |
| Power socket | 2.5 sq.mm | 16A |
| AC (1.5 ton) | 4 sq.mm | 20A |
| Geyser | 4 sq.mm | 25A |
| Main line | As per load | MCCB |

### Conduit & Wiring:

**1. Concealed Conduit:**
- Min. size: 20mm (single circuit), 25mm (multiple)
- Max. cables: 40% fill ratio
- Chase depth: conduit + 10mm
- Bends: max 4 nos between boxes
- Junction boxes at changes

**2. Point Heights:**
| Point | Height from FFL |
|-------|----------------|
| Switch board | 1200mm |
| Power socket | 300mm |
| AC point | 2100mm (or as per unit) |
| Geyser point | 1800mm |
| Exhaust fan | 2100mm |
| Bell push | 1200mm |

### Earthing Requirements:
- Pipe earthing or plate earthing
- Earth resistance: < 1 ohm
- All metal parts to be earthed
- Earth wire: Green/Yellow stripe
- Size: Same as phase wire (min 2.5 sq.mm)

### Quality Checkpoints:
- [ ] Conduit runs as per drawing
- [ ] No conduit kinks or sharp bends
- [ ] Draw wire inserted
- [ ] Junction boxes accessible
- [ ] Proper fire stopping at penetrations
- [ ] Megger test passed (min 1 MΩ)
- [ ] Earthing resistance < 1 ohm
- [ ] Phase-neutral polarity correct

### Testing:
| Test | Requirement |
|------|-------------|
| Insulation resistance | > 1 MΩ |
| Earth continuity | < 1 Ω |
| Polarity | Correct L-N-E |
| ELCB/RCCB | Trip at 30mA |

### Color Coding (India):
| Wire | Single Phase | Three Phase |
|------|-------------|-------------|
| Phase | Red | R-Y-B |
| Neutral | Black | Black |
| Earth | Green/Yellow | Green/Yellow |

### Common Defects:
| Defect | Cause | Prevention |
|--------|-------|------------|
| Short circuit | Damaged insulation | Proper conduit, draw carefully |
| Earth fault | Poor earthing | Proper earth electrode |
| Overload | Undersized cable | Calculate load, proper sizing |

**Reference:** IS 732 (Wiring), IS 3043 (Earthing), National Electrical Code
"""

    def _handle_doors_windows_query(self, question: str, context: dict) -> str:
        """Handle door and window queries."""
        return """**Answer for Site Engineer:**

## Doors & Windows - Complete Guide

### Standard Sizes:
| Type | Width | Height |
|------|-------|--------|
| Main door | 1050-1200mm | 2100mm |
| Bedroom door | 900mm | 2100mm |
| Bathroom door | 750mm | 2000mm |
| Window (bedroom) | 1200mm | 1200mm |
| Window (toilet) | 600mm | 450mm |
| Ventilator | 600mm | 300mm |

### Frame Materials:

| Material | Use | Advantages |
|----------|-----|------------|
| Wooden | Traditional, premium | Aesthetic, insulation |
| Steel | Main doors, commercial | Security, durability |
| Aluminum | Windows, sliding | Lightweight, maintenance-free |
| uPVC | Windows, doors | Thermal insulation, waterproof |

### Installation Process:

**1. Frame Fixing:**
- Check opening size (+10mm on each side)
- Fix frame with hold-fasts (min 3 per jamb)
- Check plumb and level
- Wedge temporarily
- Fill gaps with mortar/foam
- Remove wedges after setting

**2. Shutter/Glass Fixing:**
- Fix hinges at 150mm from top/bottom
- Third hinge at center for tall doors
- Check swing clearance
- Fix locks, handles
- Adjust for smooth operation

### Quality Checkpoints:
- [ ] Frame size matches opening
- [ ] Plumb and level verified
- [ ] Hold-fasts/anchors secure
- [ ] No gaps between frame and wall
- [ ] Shutters swing freely
- [ ] Locks operate smoothly
- [ ] Glass properly seated (with putty/gasket)
- [ ] Hardware complete and functional

### Clearances:
| Location | Clearance |
|----------|-----------|
| Floor (inside) | 6-10mm |
| Top/sides | 3-5mm |
| Meeting stiles | 3-5mm |
| Door bottom (bathroom) | 15-20mm |

### Common Defects:
| Defect | Cause | Prevention |
|--------|-------|------------|
| Frame out of plumb | Poor installation | Check with spirit level |
| Shutter sagging | Weak hinges | Use proper size hinges |
| Water ingress | No sealant | Apply silicone sealant |
| Rattling | Loose fit | Adjust striker plate |

**Reference:** IS 4021 (Timber), IS 1038 (Steel), IS 1948 (Aluminum)
"""

    def _handle_concrete_query(self, question: str, context: dict) -> str:
        """Handle general concrete queries."""
        return """**Answer for Site Engineer:**

## Concrete Work - Complete Guide

### Concrete Grades & Uses:
| Grade | Strength (MPa) | Use |
|-------|----------------|-----|
| M10 | 10 | Lean concrete, leveling |
| M15 | 15 | PCC, flooring |
| M20 | 20 | General RCC (mild exposure) |
| M25 | 25 | RCC (moderate exposure) |
| M30 | 30 | RCC (severe exposure) |
| M35 | 35 | Post-tensioned, precast |
| M40+ | 40+ | Special structures |

### Concreting Process:

**1. Pre-Concreting:**
- Check formwork, reinforcement
- Verify cover blocks
- Ensure cube moulds ready
- Wet formwork (not waterlogged)

**2. During Concreting:**
- Check slump at delivery
- Pour within 2 hours of batching
- Layer thickness: max 450mm
- Vibrate 10-15 sec per point
- Avoid cold joints (pour within 30 min)

**3. Post-Concreting:**
- Start curing within 24 hours
- Cure for 7-14 days based on cement
- Deshuttering as per schedule
- Document any defects

### Mixing & Placing:
| Aspect | Requirement |
|--------|-------------|
| Free fall height | Max 1.5m |
| Transit time | Max 2 hours |
| Slump (general) | 75-100mm |
| Vibration radius | 300-450mm |
| Revolutions in mixer | 20-25 |

### Cube Testing:
- Sample: 1 per 15 m³ or fraction
- Minimum: 4 samples per day
- Test at: 7 days, 28 days
- Acceptance: As per IS 456 Table 11

### Deshuttering Time (OPC, 20°C):
| Member | Days |
|--------|------|
| Vertical (walls/columns) | 16-24 hrs |
| Slab soffit (props left) | 3 days |
| Beam soffit (props left) | 7 days |
| Props (slab < 4.5m) | 7 days |
| Props (beam < 6m) | 14 days |

**Reference:** IS 456:2000, IS 10262:2019
"""

    def _handle_safety_query(self, question: str, context: dict) -> str:
        """Handle safety related queries."""
        return """**Answer for Site Engineer:**

## Construction Safety - Essential Guide

### PPE Requirements:
| PPE | Mandatory For |
|-----|---------------|
| Safety helmet | All site personnel |
| Safety shoes | All workers |
| Safety goggles | Grinding, cutting, chipping |
| Gloves | Handling materials, chemicals |
| Safety harness | Work at height > 2m |
| Ear plugs | High noise areas |
| Dust mask | Dusty operations |
| Reflective vest | Traffic areas |

### Scaffolding Safety:
- [ ] Certified scaffolding material
- [ ] Base plates on firm ground
- [ ] Bracing at all levels
- [ ] Guard rails at 1m height
- [ ] Toe boards provided
- [ ] Access ladder secured
- [ ] Max load displayed
- [ ] Daily inspection tag

### Fall Protection (Work at Height):
- Harness for work above 2m
- Anchor point: rated for 22 kN
- Full body harness preferred
- Lifeline if mobility needed
- Barricade open edges
- Safety nets where applicable

### Excavation Safety:
- Shoring for depth > 1.2m
- Barricading around excavation
- Material 1m away from edge
- Safe access (ladder/ramp)
- Dewatering if needed
- No workers under suspended loads

### Fire Safety:
- Fire extinguisher types: ABC, CO2
- Location: 15m travel distance
- Hot work permit required
- Fire watch for 30 min after hot work

### Electrical Safety:
- GFCI/ELCB for temporary power
- No open/damaged cables
- Proper grounding
- Lockout/tagout for maintenance

**Reference:** IS 3764 (Safety code), BOCW Act
"""

    def _handle_earthwork_query(self, question: str, context: dict) -> str:
        """Handle earthwork and excavation queries."""
        return """**Answer for Site Engineer:**

## Earthwork & Excavation - Complete Guide

### Soil Classification:
| Type | Description | Angle of Repose |
|------|-------------|-----------------|
| Hard Rock | Requires blasting | 90° |
| Soft Rock | Picks/breakers | 80° |
| Hard Soil | Difficult excavation | 45-60° |
| Ordinary Soil | Spade/JCB | 30-45° |
| Sandy/Loose | Easy excavation | 25-35° |

### Excavation Process:

**1. Before Excavation:**
- Identify underground utilities
- Mark excavation boundaries
- Set benchmarks
- Plan for dewatering
- Arrange shoring materials

**2. During Excavation:**
- Slope sides or shore as needed
- Keep spoil 1m from edge
- Provide safe access
- Check levels frequently
- Dewater if required

**3. For Foundation:**
- Excavate to design level
- Check bearing capacity
- Level bottom of pit
- Provide PCC immediately

### Shoring Requirements:
| Depth | Requirement |
|-------|-------------|
| < 1.2m | Sloping may suffice |
| 1.2-3m | Sheet piling/shoring |
| > 3m | Engineered shoring |

### Backfilling:
- Use approved fill material
- Layer thickness: 150-300mm
- Compact each layer
- Achieve 95% MDD (Proctor)
- Test compaction by sand replacement

### Quality Checkpoints:
- [ ] Excavation to correct level
- [ ] Side slopes safe or shored
- [ ] No water accumulation
- [ ] Bearing capacity verified
- [ ] PCC laid immediately
- [ ] Backfill material approved
- [ ] Compaction test passed

### Compaction Standards:
- Relative compaction: 95% MDD
- Test frequency: Every 100 m³ or lift
- Test method: Sand replacement or core cutter

**Reference:** IS 3764 (Safety), IS 2720 (Soil testing)
"""

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

    def _handle_kb_query(self, question: str, context: dict, kb_matches: Dict[str, List[str]]) -> str:
        """Create a response based on which KB topics matched the question."""
        parts = ["**Answer (based on internal IS Code Knowledge Base)**\n\n"]
        parts.append(f"Your question: **{question}**\n\n")

        # Summarize IS code matches
        if kb_matches.get("is_codes"):
            parts.append("**Relevant IS Codes:**\n")
            for code in kb_matches["is_codes"]:
                code_spec = self.knowledge_base.get("is_codes", {}).get(code, {})
                parts.append(f"- {code}: {code_spec.get('title', '')}\n")
                # Short sample from key_provisions
                prov = code_spec.get('key_provisions', {})
                if prov:
                    parts.append("  Key provisions:\n")
                    # Show a couple of relevant keys
                    for k, v in list(prov.items())[:3]:
                        parts.append(f"  - {k}: {str(v)}\n")

        if kb_matches.get('issues'):
            parts.append("**Known Quality Issues:**\n")
            for issue in kb_matches['issues']:
                issue_info = self.knowledge_base.get('quality_issues', {}).get(issue, {})
                parts.append(f"- {issue}: {issue_info.get('description', '')}\n")

        if kb_matches.get('checklists'):
            parts.append("**Applicable Checklists:**\n")
            for ck in kb_matches['checklists']:
                checklist = self.knowledge_base.get('checklists', {}).get(ck, {})
                parts.append(f"- {checklist.get('title', ck)}\n")

        if kb_matches.get('materials'):
            parts.append("**Material Specs:**\n")
            for m in kb_matches['materials']:
                mat = self.knowledge_base.get('materials', {}).get(m, {})
                parts.append(f"- {m}: {mat.get('types', '')}\n")

        # Add topic-based guidance
        if kb_matches.get('topics'):
            parts.append("\n**Topic Guidance:**\n")
            for tk in kb_matches.get('topics'):
                topic = self.knowledge_base.get('topic_guides', {}).get(tk, {})
                parts.append(f"- {topic.get('title', tk)}: {topic.get('summary','')}\n")
                parts.append("  Key points:\n")
                for kp in topic.get('key_points', [])[:5]:
                    parts.append(f"   - {kp}\n")

        parts.append("\nIf you'd like a concise checklist or a specific guidance point, please mention the member (slab/beam/etc.) and the grade/exposure if applicable.")

        # Add web sources if regionally present to enrich the answer
        web_ctx = context.get('web_context', [])
        if web_ctx:
            parts.append("\nSources Considered (web results):\n")
            for w in web_ctx[:5]:
                parts.append(f"- {w.get('title', '')} ({w.get('url', '')})\n")

        return "\n".join(parts)

    def _handle_kb_index(self) -> str:
        """Return a neat text list of KB IS codes and Topics."""
        idx = self.get_kb_index()
        lines = ["**Internal Knowledge Base - Index**\n"]
        lines.append("**IS Codes:**\n")
        for c in idx.get('is_codes', []):
            lines.append(f"- {c}\n")
        lines.append("\n**Topics:**\n")
        for t in idx.get('topics', []):
            lines.append(f"- {t}\n")
        lines.append("\n**Checklists:**\n")
        for ck in idx.get('checklists', []):
            lines.append(f"- {ck}\n")
        lines.append('\nUse "Is this in KB?" or ask any specific topic to get more detail.')
        return "\n".join(lines)
    
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

