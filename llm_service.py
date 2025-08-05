"""
LM Studio LLM Service for OrganicGuard AI
Provides advanced AI chat functionality using LM Studio's local models
"""

import requests
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OrganicGuardLLM:
    """LM Studio LLM service for organic pest management conversations"""
    
    def __init__(self, base_url: str = "http://localhost:1234", model_name: str = None):
        """
        Initialize the LM Studio LLM service
        
        Args:
            base_url: LM Studio server URL (default: http://localhost:1234)
            model_name: Specific model to use (optional, uses default loaded model)
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.available = False
        self.timeout = 60   
        
        # Test connection on initialization
        self._test_connection()
        
        # System prompt for organic pest management context
        self.system_prompt = """You are OrganicGuard AI, an expert organic pest management assistant. You specialize in:

1. Identifying pests from descriptions
2. Recommending organic, eco-friendly treatment methods
3. Providing IPM (Integrated Pest Management) strategies
4. Explaining prevention techniques
5. Offering cost-effective organic solutions
6. Ensuring treatments are safe for beneficial insects and the environment

Guidelines:
- Always prioritize organic, chemical-free solutions
- Consider beneficial insects and pollinators
- Provide practical, actionable advice
- Include timing and application instructions
- Mention cost considerations when relevant
- Keep responses focused, helpful and summarised
- If unsure, recommend consulting local extension services

Respond in a helpful, knowledgeable tone suitable for both hobbyist gardeners and professional organic farmers."""

    def _test_connection(self) -> bool:
        """Test connection to LM Studio server"""
        try:
            # First try to connect to the base URL
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=5
            )
            if response.status_code == 200:
                self.available = True
                models = response.json()
                model_count = len(models.get('data', []))
                logger.info(f"✅ Connected to LM Studio. Available models: {model_count}")
                
                if model_count == 0:
                    logger.warning("⚠️ LM Studio connected but no models loaded!")
                    logger.warning("   Please load a model in LM Studio to use chat features")
                
                return True
            elif response.status_code == 404:
                logger.warning(f"⚠️ LM Studio API not found (404). Check if:")
                logger.warning(f"   1. LM Studio is running on {self.base_url}")
                logger.warning(f"   2. The local server is enabled in LM Studio")
                logger.warning(f"   3. Try different port if not using default 1234")
                return False
            else:
                logger.warning(f"⚠️ LM Studio responded with status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Cannot connect to LM Studio at {self.base_url}")
            logger.warning("   Make sure LM Studio is running and local server is enabled")
            self.available = False
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Connection error: {e}")
            self.available = False
            return False

    def generate_response(self, user_message: str, conversation_history: Optional[list] = None) -> str:
        """
        Generate response using LM Studio model
        
        Args:
            user_message: User's question or message
            conversation_history: Previous conversation context (optional)
            
        Returns:
            AI-generated response string
        """
        if not self.available:
            logger.warning("LM Studio not available, using fallback response")
            return self._generate_fallback_response(user_message)
        
        try:
            # Prepare messages for the API
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Prepare request payload
            payload = {
                "model": self.model_name or "local-model",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False,
                "stop": None
            }
            
            # Make request to LM Studio
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'choices' in data and len(data['choices']) > 0:
                    ai_response = data['choices'][0]['message']['content'].strip()
                    logger.info(f"✅ LM Studio response generated ({len(ai_response)} chars)")
                    return ai_response
                else:
                    logger.error("❌ Invalid response format from LM Studio")
                    return self._generate_fallback_response(user_message)
            else:
                logger.error(f"❌ LM Studio API error: {response.status_code} - {response.text}")
                if response.status_code == 404:
                    logger.error("🚨 404 Error: LM Studio server not found. Check if:")
                    logger.error("   1. LM Studio is running")
                    logger.error("   2. Server is on http://localhost:1234")
                    logger.error("   3. A model is loaded in LM Studio")
                return self._generate_fallback_response(user_message)
                
        except requests.exceptions.Timeout:
            logger.error("❌ LM Studio request timed out")
            return "I'm experiencing some delays. Here's a quick tip: For most organic pest issues, neem oil and insecticidal soap are excellent starting points. What specific pest are you dealing with?"
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ LM Studio request failed: {e}")
            return self._generate_fallback_response(user_message)
        
        except Exception as e:
            logger.error(f"❌ Unexpected error in LM Studio service: {e}")
            return self._generate_fallback_response(user_message)

    def _generate_fallback_response(self, user_message: str) -> str:
        """
        Generate specialized fallback responses when LM Studio is unavailable
        
        Args:
            user_message: User's message to respond to
            
        Returns:
            Specialized organic pest management response
        """
        message = user_message.lower()
        
        # Pest-specific responses
        if any(word in message for word in ['aphid', 'aphids']):
            return """🐛 **Aphid Control (Organic Methods)**

**Immediate Actions:**
• Insecticidal soap: 2 tbsp mild soap per gallon water
• Neem oil spray: 2-4 tbsp per gallon, apply early morning/evening
• Strong water spray to dislodge colonies

**Biological Control:**
• Release ladybugs (1,500 per small garden)
• Attract lacewings with diverse flowering plants
• Plant companion herbs: basil, catnip, garlic

**Prevention:**
• Avoid over-fertilizing with nitrogen
• Use reflective mulches to confuse aphids
• Regular monitoring of undersides of leaves

💰 **Cost**: $15-25 for small garden treatment
⏰ **Best timing**: Early morning application, reapply weekly"""

        elif any(word in message for word in ['spider', 'mite', 'mites']):
            return """🕷️ **Spider Mite Management (Organic)**

**Symptoms**: Fine webbing, stippled leaves, bronze coloring

**Organic Treatments:**
• Predatory mites (Phytoseiulus persimilis)
• Insecticidal soap + neem oil combination
• Increase humidity around plants
• Strong water sprays 3x weekly

**Cultural Controls:**
• Avoid dusty conditions (mites thrive in dust)
• Proper plant spacing for air circulation
• Remove heavily infested leaves

**Prevention:**
• Regular misting in dry conditions
• Companion planting with aromatic herbs
• Beneficial insect habitat (diverse plantings)

⚠️ **Important**: Spider mites reproduce rapidly in hot, dry conditions"""

        elif any(word in message for word in ['slug', 'snail', 'slugs', 'snails']):
            return """🐌 **Slug & Snail Control (Organic)**

**Physical Barriers:**
• Copper tape around containers/beds
• Diatomaceous earth rings (reapply after rain)
• Sharp gravel or crushed eggshells

**Organic Baits:**
• Iron phosphate pellets (pet/wildlife safe)
• Beer traps (shallow dishes, replace every 2-3 days)
• Grapefruit rinds as traps (collect daily)

**Habitat Modification:**
• Remove hiding spots (boards, debris, thick mulch)
• Water in morning only (not evening)
• Create dry zones around sensitive plants

**Natural Predators:**
• Encourage birds with bird baths/houses
• Ground beetles (provide log shelters)
• Garter snakes (if climate appropriate)

🌙 **Best collection time**: Early morning or with flashlight at night"""

        elif any(word in message for word in ['whitefly', 'whiteflies']):
            return """🦟 **Whitefly Management (Organic)**

**Immediate Control:**
• Yellow sticky traps (replace weekly)
• Insecticidal soap spray (2-3x weekly)
• Neem oil (systemic action, 7-14 day intervals)

**Biological Control:**
• Encarsia wasps (parasitize whitefly nymphs)
• Ladybugs for adult feeding
• Reflective mulches to confuse adults

**Cultural Practices:**
• Remove yellowing leaves promptly
• Avoid over-fertilizing (makes plants attractive)
• Companion planting: marigolds, nasturtiums
• Proper plant spacing for air circulation

**Monitoring:**
• Check undersides of leaves weekly
• Shake plants gently - adults will fly if present
• Yellow sticky trap counts for population tracking

⚠️ **Key**: Whiteflies have rapid reproduction cycles - consistency is crucial"""

        elif any(word in message for word in ['caterpillar', 'catterpillar', 'worm', 'hornworm']):
            return """🐛 **Caterpillar Control (Organic Methods)**

**Bt Spray (Bacillus thuringiensis):**
• Apply in evening (UV sensitive)
• Target young caterpillars (more effective)
• Reapply after rain, every 7-10 days
• Safe for humans, pets, beneficial insects

**Physical Removal:**
• Hand-picking (early morning most effective)
• Look for frass (droppings) to locate them
• Check undersides of leaves regularly
• Use flashlight for night inspection

**Natural Predators:**
• Encourage birds with houses/water
• Beneficial wasps (plant diverse flowers)
• Spiders and ground beetles
• Chickens if garden setup allows

**Prevention:**
• Row covers during egg-laying periods
• Companion planting: dill, fennel (attract beneficial wasps)
• Regular garden cleanup
• Crop rotation annually

🔍 **Detection tip**: Look for chewed leaf edges and dark frass on leaves"""

        elif any(word in message for word in ['scale', 'scales']):
            return """🛡️ **Scale Insect Control (Organic)**

**Oil Treatments:**
• Horticultural oil (2-4 tbsp per gallon)
• Neem oil (systemic action)
• Apply during cooler parts of day
• Ensure complete coverage of stems/leaves

**Physical Removal:**
• Soft brush + soapy water for light infestations
• Fingernail or soft scraper for individuals
• Rubbing alcohol on cotton swab for spot treatment

**Systemic Approach:**
• Beneficial insects: ladybugs, lacewings
• Parasitic wasps (Aphytis, Encarsia species)
• Improve plant health (proper fertilization/watering)

**Prevention:**
• Quarantine new plants 2-3 weeks
• Regular inspection (scales are sneaky!)
• Avoid over-fertilizing with nitrogen
• Proper plant spacing for air circulation

⏰ **Best treatment time**: During crawler stage (mobile phase)"""

        elif any(word in message for word in ['thrips']):
            return """⚡ **Thrips Management (Organic)**

**Identification:**
• Tiny (1-2mm), slender insects
• Silver/bronze leaf streaks
• Black specks (excrement) on leaves
• Flowers may be deformed/discolored

**Organic Controls:**
• Blue sticky traps (thrips prefer blue over yellow)
• Insecticidal soap + neem oil combination
• Predatory mites (Amblyseius species)
• Beneficial nematodes for soil-pupating species

**Cultural Practices:**
• Remove weeds (alternate hosts)
• Reflective mulches deter adults
• Proper irrigation (thrips prefer stressed plants)
• Remove spent flowers/damaged leaves

**Biological Control:**
• Minute pirate bugs (Orius species)
• Lacewing larvae
• Predatory thrips species
• Encourage with diverse flowering plants

💡 **Pro tip**: Thrips are most active in warm, dry conditions"""

        elif any(word in message for word in ['fungus', 'mold', 'mildew', 'blight']):
            return """🍄 **Organic Fungal Disease Management**

**Preventive Sprays:**
• Baking soda spray: 1 tsp per quart water + drop of soap
• Milk spray: 1 part milk to 9 parts water
• Compost tea: Weekly application
• Neem oil: Also prevents fungal issues

**Cultural Controls:**
• Improve air circulation (proper spacing)
• Water at soil level (avoid wetting leaves)
• Morning watering (leaves dry quickly)
• Remove infected plant material immediately

**Soil Health:**
• Add compost regularly (beneficial microorganisms)
• Ensure proper drainage
• Mulch to prevent soil splash onto leaves
• Crop rotation to break disease cycles

**Resistant Varieties:**
• Choose disease-resistant cultivars when possible
• Native plants often have better natural resistance
• Proper plant selection for your climate

⚠️ **Remember**: Prevention is much easier than treatment for fungal issues"""

        # General responses
        elif any(word in message for word in ['organic', 'natural', 'safe']):
            return """🌱 **Core Organic Pest Management Principles**

**The Organic Toolbox:**
1. **Biological Control**: Beneficial insects, parasites, predators
2. **Cultural Practices**: Crop rotation, companion planting, sanitation
3. **Physical Barriers**: Row covers, traps, barriers
4. **Organic Sprays**: Neem oil, insecticidal soap, botanical oils
5. **Soil Health**: Compost, beneficial microorganisms

**Safety First:**
• Always read labels, even on organic products
• Apply treatments during cooler parts of day
• Avoid spraying during peak pollinator activity
• Wear appropriate protective equipment

**Cost-Effective Strategies:**
• Prevention costs less than treatment
• Make your own insecticidal soap
• Encourage beneficial insects (free labor!)
• Build healthy soil for natural plant resistance

**Certification Compatible:**
Most methods mentioned are approved for organic certification, but always verify with your certifying body if commercial."""

        elif any(word in message for word in ['beneficial', 'insects', 'predator', 'parasites']):
            return """🐞 **Beneficial Insects for Organic Pest Control**

**Key Beneficial Species:**
• **Ladybugs**: Aphids, mites, small caterpillars
• **Lacewings**: Aphids, thrips, whiteflies, mites
• **Parasitic wasps**: Various pest species (host-specific)
• **Predatory mites**: Spider mites, thrips
• **Ground beetles**: Slugs, caterpillars, soil pests

**How to Attract Beneficials:**
• Diverse flowering plants (succession blooming)
• Native plants and wildflower areas
• Shallow water sources
• Shelter options (logs, stones, diverse plantings)
• Reduce pesticide use (even organic ones when beneficial insects present)

**Commercial Releases:**
• Best for greenhouse/protected environments
• Release when pest populations are moderate
• Follow supplier instructions for timing/numbers
• Provide food sources (flowers) and shelter

**Plants That Attract Beneficials:**
• Yarrow, dill, fennel, sweet alyssum
• Marigolds, nasturtiums, sunflowers
• Native wildflowers and herbs

🌼 **Pro tip**: A diverse ecosystem naturally balances pest and beneficial populations"""

        elif any(word in message for word in ['cost', 'budget', 'expensive', 'cheap']):
            return """💰 **Budget-Friendly Organic Pest Management**

**DIY Solutions (Under $10):**
• Insecticidal soap: 2 tbsp dish soap + 1 gallon water
• Garlic/pepper spray: Blend + strain + spray
• Beer traps for slugs: Use shallow containers
• Companion planting: Seeds cost less than treatments

**Mid-Range Options ($10-50):**
• Neem oil concentrate (multiple applications)
• Beneficial insect releases for small gardens
• Row covers (reusable for several seasons)
• Organic certified products

**Investment Options ($50+):**
• Predatory mite releases for larger areas
• Quality organic fertilizers (long-term plant health)
• Copper barriers for permanent slug control
• Professional beneficial insect habitat plants

**Money-Saving Tips:**
• Prevention always costs less than treatment
• Buy neem oil concentrate vs. ready-to-use
• Share beneficial insect orders with neighbors
• Start with cheapest effective option first

📊 **ROI Focus**: Healthy soil + diverse plantings = fewer pest problems long-term"""

        elif any(word in message for word in ['timing', 'when', 'schedule']):
            return """⏰ **Optimal Timing for Organic Treatments**

**Daily Timing:**
• **Early morning (6-8 AM)**: Cool temperatures, calm winds
• **Evening (6-8 PM)**: After pollinators are less active
• **Avoid midday**: Heat stress on plants, active beneficial insects

**Seasonal Considerations:**
• **Spring**: Preventive treatments, beneficial releases
• **Summer**: Regular monitoring, targeted treatments
• **Fall**: Garden cleanup, soil preparation
• **Winter**: Planning, equipment maintenance

**Weather Factors:**
• No rain expected for 4-6 hours minimum
• Light winds (under 10 mph for spray applications)
• Temperature between 65-85°F for most treatments
• Higher humidity helps some beneficial microorganisms

**Pest Life Cycle Timing:**
• Target vulnerable stages (young caterpillars, crawler scales)
• Monitor for peak activity periods
• Coordinate treatments with natural enemy releases

**Treatment Schedules:**
• Weekly monitoring walks
• Bi-weekly preventive treatments if needed
• Monthly soil health assessments

🌡️ **Weather apps** are your friend for timing applications!"""

        else:
            return """🌿 **OrganicGuard AI - Your Organic Pest Management Assistant**

I'm here to help with organic, eco-friendly pest management! I can assist you with:

**🔍 Pest Identification & Treatment:**
• Upload photos for AI identification
• Specific organic treatment recommendations
• Safe application methods and timing

**🌱 Sustainable Approaches:**
• Integrated Pest Management (IPM) strategies
• Beneficial insect attraction and conservation
• Soil health for natural plant resistance

**💡 Practical Guidance:**
• Cost-effective solutions for any budget
• Prevention strategies (cheaper than treatment!)
• Organic certification-compatible methods

**Popular Topics:**
• "How do I control aphids organically?"
• "What are safe treatments for caterpillars?"
• "How can I attract beneficial insects?"
• "Organic solutions for slugs and snails?"
• "Budget-friendly pest management tips?"

**📸 For best results**: Upload a clear photo of the pest or damage, or describe your specific situation!

What pest challenge can I help you solve today?"""

    def set_custom_system_prompt(self, prompt: str):
        """Set a custom system prompt for specialized conversations"""
        self.system_prompt = prompt
        logger.info("Custom system prompt set for LM Studio service")

    def get_available_models(self) -> list:
        """Get list of available models from LM Studio"""
        if not self.available:
            return []
        
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
            return []
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status information"""
        return {
            'available': self.available,
            'base_url': self.base_url,
            'model_name': self.model_name,
            'models_available': len(self.get_available_models()),
            'connection_test': self._test_connection()
        }
