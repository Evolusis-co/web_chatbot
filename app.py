"""
Simple Web Chatbot - Uses Qdrant for context retrieval with OpenAI embeddings
"""

import os
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
import logging
from datetime import datetime

# Load environment variables (don't override existing ones from Render)
load_dotenv(override=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

# Configure CORS for specific origins from environment variable
cors_origins = os.getenv("CORS_ORIGINS", "").split(",")
CORS(app, 
     origins=cors_origins if cors_origins[0] else ["*"],
     supports_credentials=True)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "bridgetext_scenarios"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global variables
qdrant_client = None
openai_client = None

def initialize_services():
    """Initialize Qdrant and OpenAI services"""
    global qdrant_client, openai_client
    
    try:
        logger.info("🔌 Connecting to services...")
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        # Initialize OpenAI client (for embeddings AND chat)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        logger.info("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        return False

def get_relevant_context(user_message: str, top_k: int = 3) -> str:
    """Retrieve relevant context from Qdrant using OpenAI embeddings"""
    try:
        # Generate embedding using OpenAI (text-embedding-3-small model)
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=user_message
        )
        query_vector = embedding_response.data[0].embedding
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        
        # Extract context from results
        context_parts = []
        for result in search_results:
            if hasattr(result, 'payload') and result.payload:
                text = result.payload.get('text', '') or result.payload.get('page_content', '')
                if text:
                    context_parts.append(text)
        
        return "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return "No context available."

def generate_response(user_message: str, context: str, chat_history: str = "", tone: str = "Professional", chat_length: int = 0) -> str:
    """Generate response using GPT-4o-mini with STEP + 4Rs framework and Qdrant context"""
    try:
        # Check if this is a greeting (first message)
        greeting_words = ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo', 'helo', 'hola']
        if user_message.lower().strip() in greeting_words and chat_length == 1:
            # Return friendly, natural greeting based on tone
            if tone == "Casual":
                return "Hey! What's up? What's going on at work?"
            else:
                return "Hello! How can I help you with your workplace challenges today?"
        
        # Safety check - Physical violence/abuse (CRITICAL) - Only if it's clearly physical violence
        # Improved: Check for context to avoid false positives (e.g., "beat me in workload")
        violence_keywords = ['hit', 'punch', 'slap', 'kick', 'physical violence', 'physically hurt', 'assault', 'attack', 'threatened with violence']
        workload_context = ['workload', 'work load', 'tasks', 'deadline', 'pressure', 'stress', 'overwhelm']
        
        # Only trigger violence warning if violence keywords found AND no workload context
        has_violence_keyword = any(keyword in user_message.lower() for keyword in violence_keywords)
        has_workload_context = any(keyword in user_message.lower() for keyword in workload_context)
        
        # Special check for "beat" - only warn if it's clearly physical, not metaphorical
        if 'beat' in user_message.lower() and not has_workload_context:
            # Check if it's physical violence context
            physical_indicators = ['physically', 'hit me', 'hurt me', 'threatened', 'violence']
            if any(indicator in user_message.lower() for indicator in physical_indicators):
                return """⚠️ **This is serious.** Physical violence at work is illegal and unacceptable.

Please take action immediately:
• Document everything (dates, witnesses, injuries)
• Report to HR or higher management NOW
• Contact workplace violence hotline: 1-800-799-7233
• If you're in immediate danger, call 911

This isn't a communication issue — it's workplace abuse. I can't coach you through this, but I strongly urge you to protect yourself and report this."""
        
        # Regular violence keywords (excluding 'beat' which is handled above)
        if has_violence_keyword and not has_workload_context:
            return """⚠️ **This is serious.** Physical violence at work is illegal and unacceptable.

Please take action immediately:
• Document everything (dates, witnesses, injuries)
• Report to HR or higher management NOW
• Contact workplace violence hotline: 1-800-799-7233
• If you're in immediate danger, call 911

This isn't a communication issue — it's workplace abuse. I can't coach you through this, but I strongly urge you to protect yourself and report this."""
        
        # Safety check - Harmful content
        harmful_keywords = ['kill', 'murder', 'suicide', 'weapon', 'gun', 'knife', 'blood', 'stab', 'threat', 'harass']
        if any(keyword in user_message.lower() for keyword in harmful_keywords):
            return """⚠️ I'm concerned about what you've shared. If you're in immediate danger or witnessing illegal activity, please contact:

• Emergency Services: 911
• National Suicide Prevention Lifeline: 988
• Workplace Violence Hotline: 1-800-799-7233

I'm designed to help with workplace communication challenges, not crisis or safety situations. Please reach out to professionals who can provide proper support."""
        
        # Safety check - Health issues
        health_keywords = ['headache', 'sick', 'pain', 'fever', 'medication', 'doctor', 'hospital']
        if any(keyword in user_message.lower() for keyword in health_keywords):
            return "I'm specifically designed for workplace communication challenges. For health concerns, please consult a medical professional. Can we focus on a work-related communication or teamwork challenge instead?"
        
        # DON'T ask for tone preference here - let the button logic handle it
        # This allows for more natural conversation flow
        
        # Special handling for tone selection - Respond to their ORIGINAL problem
        if user_message.strip() in ["Professional", "Casual"]:
            selected_tone = user_message.strip()
            
            # Get the user's problem from chat history
            if chat_history:
                # Extract all user messages from chat history
                history_lines = chat_history.strip().split('\n')
                user_messages = []
                for line in history_lines:
                    if line.startswith('User:'):
                        msg = line.replace('User:', '').strip()
                        # Skip greetings and tone selections
                        if msg.lower() not in ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo', 'professional', 'casual']:
                            user_messages.append(msg)
                
                # If we found their problem, REPLACE user_message with it
                if user_messages:
                    user_problem = user_messages[-1]  # Get the most recent problem statement
                    # CRITICAL: Replace "Casual" with their actual problem so GPT responds to THAT
                    user_message = user_problem
                    logger.info(f"🔄 Tone selected: {selected_tone}. Responding to original problem: {user_problem[:50]}...")
        
        # Define tone-specific instructions
        if tone == "Casual":
            tone_instruction = """⚠️ MANDATORY CASUAL TONE - DO NOT USE FORMAL LANGUAGE:
• Sound like texting a smart friend - relaxed, Gen Z, conversational
• MUST use phrases like: "That sucks", "Ugh that's rough", "Yeah I totally get it", "Super frustrating"
• ALWAYS use contractions: "you're", "that's", "don't", "can't", "it's"
• Keep it SHORT and REAL - like you're texting, NOT writing an essay
• Be supportive but chill: "Okay let's figure this out" NOT "I understand your concern"
• NEVER say: "I appreciate", "However", "It's important to maintain", "effectively address"
• Example CORRECT casual: "Ugh that really sucks. Sounds like you're feeling burned out? What's making you wanna quit - the work itself or something else?"
"""
        else:  # Professional
            tone_instruction = """⚠️ MANDATORY PROFESSIONAL TONE - DO NOT USE SLANG:
• Sound like a workplace mentor or HR coach - measured, empathetic, formal
• Use complete sentences with proper grammar
• Use phrases like: "I understand this is challenging", "That's a difficult situation", "Let's explore this together"
• Be empathetic but maintain professional distance
• NEVER use slang or casual phrases like "sucks", "ugh", "totally"
• Example CORRECT professional: "That's a challenging situation. It sounds like communication barriers are impacting your work. Have you had an opportunity to address this directly with your colleague?"
"""
        
        system_prompt = f"""You are a Gen Z workplace coach helping young professionals with work challenges.

Use these frameworks:
• STEP (Adaptability): Spot → Think → Engage → Perform
• 4Rs (Emotional Intelligence): Recognize → Regulate → Respect → Reflect

⸻

🎯 CRITICAL FORMATTING INSTRUCTIONS:

Your response will be displayed in a web chat interface. Format it using HTML:

1. Use <b>keyword</b> for bold text (e.g., <b>Spot</b>, <b>Think</b>)
2. Use <br> for line breaks between bullet points (single break only!)
3. Use • symbol for bullets

Example format:
"Ugh that sounds rough. Here's how to handle it with STEP:<br><br>• <b>Spot</b> - Figure out which tasks are urgent<br>• <b>Think</b> - Your boss might not realize they're overloading you<br>• <b>Engage</b> - Ask for a quick 15-min priority check<br>• <b>Perform</b> - Track progress and escalate if needed<br><br>Sound good?"

KEY RULES:
- NEVER use **text** for bold, use <b>text</b> instead
- NEVER use \\n for line breaks, use <br> instead
- Use <br> (single break) between bullets, <br><br> (double break) only after intro text
- NEVER use numbered lists (1. 2. 3.), always use bullets (•)
- Keep total response under 120 words

⸻

🎯 TONE:
{tone_instruction}

⸻

🎯 CONVERSATION APPROACH:

**When user JUST selected their tone (Professional/Casual):**
- Look at the PREVIOUS message (before tone selection) to see what problem they described
- Respond directly to THAT problem in the selected tone
- Don't say "I understand this is challenging" - that's generic filler
- Jump straight into helping with their specific issue

**CRITICAL - ALWAYS acknowledge NEW information:**
- If user adds deadlines, urgency, constraints, or new details → ACKNOWLEDGE IT in your response!
- Example: User says "submission is within 3 days" → Respond with deadline urgency in mind
- NEVER ignore what the user just told you

**General flow:**
- First response to a problem: Ask 1 clarifying question
- Second response onwards: Give actionable STEP or 4Rs advice with bullets
- Keep responses SHORT and conversational

⸻

**CONTEXT FROM YOUR DATASET:**
{context}

**PREVIOUS CONVERSATION:**
{chat_history}

**USER JUST SAID:**
{user_message}

**YOUR RESPONSE (Use <b> for bold, <br><br> between bullets):**"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,  # Reduced from 0.7 for more consistent tone adherence
            max_tokens=200
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # POST-PROCESS: Force proper formatting if GPT didn't follow instructions
        formatted_response = format_response(raw_response)
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Sorry, I'm having trouble generating a response right now. Please try again."

def format_response(text: str) -> str:
    """Format response with proper HTML line breaks and bold text"""
    import re
    
    # Replace **text** with <b>text</b> for bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    
    # If response contains bullets, ensure proper line breaks (compact spacing!)
    if '•' in text:
        # Split by bullet points
        parts = text.split('•')
        intro = parts[0].strip()
        bullets = ['• ' + part.strip() for part in parts[1:] if part.strip()]
        
        # Rejoin with SINGLE line break only (compact!)
        if bullets:
            formatted_bullets = '<br>'.join(bullets)  # Single <br> between bullets
            text = f"{intro}<br><br>{formatted_bullets}"  # Double break only after intro
    
    # Replace numbered lists with bullets if present
    text = re.sub(r'(\d+)\.\s+\*\*([^*]+)\*\*', r'• <b>\2</b>', text)
    text = re.sub(r'(\d+)\.\s+<b>([^<]+)</b>', r'• <b>\2</b>', text)
    
    # Ensure line breaks after framework introductions (but keep it compact)
    text = re.sub(r'(using STEP:|with STEP:|STEP method:|4Rs framework:)', r'\1<br><br>', text, flags=re.IGNORECASE)
    
    # Clean up any triple or quadruple line breaks that might have slipped through
    text = re.sub(r'<br>\s*<br>\s*<br>+', '<br><br>', text)  # Max 2 breaks
    
    return text

# Initialize services on startup
initialize_services()

# Routes
@app.route('/')
def index():
    """Main chat interface"""
    # Initialize session chat history
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        original_user_message = user_message  # Save original before any modifications
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        logger.info(f"📨 User: {user_message}")
        
        # Check if services are initialized
        if not qdrant_client or not openai_client:
            logger.error("Services not initialized, attempting to reinitialize...")
            if not initialize_services():
                return jsonify({'error': 'Services unavailable. Please try again later.', 'success': False}), 503
        
        # Check message limit (10 messages = 5 exchanges)
        current_count = len(session.get('chat_history', []))
        if current_count >= 10:
            return jsonify({
                'response': "You've reached the free message limit (10 messages). Upgrade to Premium for unlimited conversations! 🚀",
                'limit_reached': True,
                'quick_replies': [],
                'success': True
            })
        
        # Get relevant context from Qdrant using OpenAI embeddings
        context = get_relevant_context(user_message)
        
        # Get chat history for context - Include MORE history (last 4 exchanges instead of 2)
        history = session.get('chat_history', [])
        chat_history = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history[-4:]])  # Last 4 exchanges for better context
        
        # Get selected tone (default to Professional)
        selected_tone = session.get('tone', 'Professional')
        
        # Calculate chat length BEFORE adding current message (for tone question detection)
        current_chat_length = len(history) + 1
        
        # Generate response using GPT-4o-mini with Qdrant context
        ai_response = generate_response(user_message, context, chat_history, selected_tone, current_chat_length)
        
        # Store in session history (use ORIGINAL message if tone was selected)
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'user': original_user_message,  # Store "Casual" or "Professional", not the replaced problem
            'ai': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        session.modified = True
        
        logger.info(f"✅ AI: {ai_response[:100]}...")
        
        # Smart quick reply flow
        is_safety_warning = ai_response.startswith("⚠️") or "call 911" in ai_response.lower()
        has_selected_tone = 'tone' in session and session['tone'] in ["Professional", "Casual"]
        chat_length = len(session.get('chat_history', []))
        
        quick_replies = []
        
        # Store tone if user just selected it
        if user_message.strip() in ["Professional", "Casual"]:
            session['tone'] = user_message.strip()
            session.modified = True
            logger.info(f"✅ Tone '{user_message.strip()}' saved to session")
        
        # Never show buttons after safety warnings
        if is_safety_warning:
            quick_replies = []
            logger.info("⚠️ Safety warning - no buttons")
        
        # Show tone selection after user describes a real problem (not just greetings)
        elif not has_selected_tone and chat_length >= 2:
            # Check if this is NOT a greeting and NOT a tone selection
            greeting_words = ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo', 'professional', 'casual']
            if user_message.lower().strip() not in greeting_words:
                # User shared a real problem - ask for tone preference
                ai_response = ai_response + "<br><br>Before we continue, how would you like me to respond? Pick your preferred tone:"
                quick_replies = ["Professional", "Casual"]
                logger.info(f"🎯 Chat #{chat_length}: Asking for tone after real query")
        
        return jsonify({
            'response': ai_response,
            'quick_replies': quick_replies,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your message.',
            'success': False
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history"""
    history = session.get('chat_history', [])
    return jsonify({'history': history})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear chat history and reset tone preference"""
    session['chat_history'] = []
    # Also clear the tone preference so user gets asked again
    if 'tone' in session:
        del session['tone']
    session.modified = True
    return jsonify({'success': True, 'message': 'Chat history cleared'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'qdrant_connected': qdrant_client is not None,
        'openai_ready': openai_client is not None,
        'model': 'gpt-4o-mini',
        'embeddings': 'text-embedding-3-small',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Web Chatbot Application")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
