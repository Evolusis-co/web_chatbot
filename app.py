"""
Simple Web Chatbot - Uses Qdrant for context retrieval with OpenAI embeddings
Token-based authentication for cross-browser compatibility
"""

import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
import logging
from datetime import datetime, timedelta
import jwt

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

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET_KEY", app.secret_key)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24  # Token valid for 24 hours

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
            api_key=QDRANT_API_KEY,
            timeout=10  # 10 second timeout
        )
        
        # Initialize OpenAI client (for embeddings AND chat) with timeout
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=30.0  # 30 second timeout for API calls
        )
        
        logger.info("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        return False

# ============================================================================
# JWT Token Functions
# ============================================================================

def create_token(chat_history: list = None, tone: str = None) -> str:
    """Create a new JWT token with chat session data"""
    payload = {
        'chat_history': chat_history or [],
        'tone': tone,
        'created_at': datetime.utcnow().isoformat(),
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_token(token: str) -> dict:
    """Decode and validate JWT token, return session data"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {
            'chat_history': payload.get('chat_history', []),
            'tone': payload.get('tone'),
            'valid': True
        }
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return {'chat_history': [], 'tone': None, 'valid': False, 'error': 'Token expired'}
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        return {'chat_history': [], 'tone': None, 'valid': False, 'error': 'Invalid token'}

def get_token_from_request() -> str:
    """Extract token from Authorization header or request body"""
    # Try Authorization header first
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]
    
    # Fallback to request body
    data = request.get_json() or {}
    return data.get('token', '')

def get_relevant_context(user_message: str, top_k: int = 3) -> str:
    """Retrieve relevant context from Qdrant using OpenAI embeddings"""
    try:
        # Skip Qdrant if not available
        if not qdrant_client:
            logger.warning("Qdrant client not initialized")
            return "No context available."
            
        # Generate embedding using OpenAI - Reduce dimensions to 768 to match Qdrant
        # text-embedding-3-small supports dimension parameter to reduce from 1536 to 768
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=user_message,
            dimensions=768  # Reduce to 768 to match Qdrant collection
        )
        query_vector = embedding_response.data[0].embedding
        
        logger.info(f"Generated embedding with {len(query_vector)} dimensions")
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        
        logger.info(f"Found {len(search_results)} results from Qdrant")
        
        # Extract context from results
        context_parts = []
        for idx, result in enumerate(search_results):
            if hasattr(result, 'payload') and result.payload:
                # Log what fields exist in the payload
                logger.info(f"Result {idx+1} payload keys: {list(result.payload.keys())}")
                
                # Try common field names for text content
                text = (result.payload.get('text') or 
                       result.payload.get('page_content') or 
                       result.payload.get('content') or
                       result.payload.get('scenario') or
                       result.payload.get('description') or '')
                
                if text:
                    context_parts.append(text)
                    logger.info(f"  ✓ Found text in result {idx+1}: {text[:100]}...")
                else:
                    logger.warning(f"  ✗ No text found in result {idx+1} payload: {result.payload}")
        
        if context_parts:
            logger.info(f"✅ Successfully retrieved {len(context_parts)} context items from Qdrant")
            return "\n\n".join(context_parts)
        else:
            logger.warning("No context found in Qdrant results")
            return "No relevant context found."
        
    except Exception as e:
        # This is a critical error - bot SHOULD use Qdrant context
        logger.error(f"❌ CRITICAL: Failed to get Qdrant context: {str(e)}")
        return "No context available."

def generate_response(user_message: str, context: str, chat_history: str = "", tone: str = None, chat_length: int = 0) -> str:
    """Generate response using GPT-4o-mini with STEP + 4Rs framework and Qdrant context"""
    try:
        # Check if this is a greeting (first message ONLY)
        greeting_words = ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo', 'helo', 'hola']
        if user_message.lower().strip() in greeting_words and chat_length <= 1:
            # Return friendly, natural greeting (no tone needed for greetings)
            return "Hello! How can I help you today?"
        
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
        
        # Define tone-specific instructions
        tone_instruction = ""
        if tone == "Casual":
            tone_instruction = """
🎯 YOU MUST USE CASUAL/FRIENDLY TONE:
- Write like you're texting a friend - use "gonna", "wanna", "that sucks", "ugh"
- NEVER use formal phrases like "I understand", "Let us", "effectively", "navigate"
- Keep it SHORT and conversational
- Use contractions always: "you're", "don't", "can't", "it's"
"""
        elif tone == "Professional":
            tone_instruction = """
🎯 YOU MUST USE PROFESSIONAL TONE:
- Sound like a workplace mentor - clear, respectful, structured
- Use complete sentences and proper grammar
- Professional but warm and supportive
"""
        else:
            # NO TONE SELECTED - ASK QUESTIONS ONLY
            tone_instruction = """
🎯 NO TONE SELECTED - YOU MUST ASK QUESTIONS:
- User hasn't picked tone yet
- DO NOT give advice or frameworks yet
- Ask 1-2 brief questions to understand their situation
- Keep it neutral and friendly
"""
        
        # Build system prompt - ONLY for when tone is selected
        if tone == "Casual":
            system_prompt = f"""You are a helpful workplace coach. NEVER mention frameworks or models.

Chat History:
{chat_history}

Current user message: "{user_message}"

🎯 TONE: Casual (like a friendly colleague)

📋 CRITICAL RULES:

**YOUR SCOPE:**
- ONLY help with workplace challenges: communication, conflicts, career growth, team dynamics, work stress
- If they ask about gossip, personal drama, or off-topic stuff → Redirect: "I'm here to help with workplace challenges. What's something work-related that's been on your mind?"
- If it's an emergency (violence, harassment, mental health crisis) → Give crisis resources, don't try to coach

**CONVERSATION FLOW:**

**FIRST RESPONSE (when problem shared):**
- Acknowledge issue briefly (1 sentence)
- If you need 1-2 key details to help, ask ONE specific question
- If situation is clear, skip to solution immediately

**AFTER 1-2 CLARIFYING QUESTIONS:**
- STOP asking questions
- Give practical advice (1-2 sentences)
- If you lack specific company info (policies, procedures), acknowledge it: "I don't have your company's specific leave policy, but here's what usually works..."
- End with simple yes/no question to keep engaged

**EXAMPLE - Leave request:**
User: "I want leave but used all my leave"
You: "That's tough. I don't have your company's specific policies, but you could try talking to your manager about unpaid leave or working from home if it's urgent. Is this for something time-sensitive?"

**EXAMPLE - Redirect gossip:**
User: "My coworker is dating the boss"
You: "I'm here to help with workplace challenges. What's something work-related that's been on your mind?"

**EXAMPLE - Emergency redirect:**
User: "I want to hurt myself"
You: "⚠️ Please reach out for immediate help: National Suicide Prevention Lifeline: 988. I'm designed for workplace challenges, not crisis support."

**STYLE:**
- Max 2-3 sentences
- Sound like a human friend: "Honestly...", "Here's what I'd try...", "You could..."
- Use contractions: "you're", "don't", "can't"
- NO robotic phrases: "It sounds like...", "I understand that...", "Thank you for sharing..."

Respond in 2-3 sentences:"""
        else:  # Professional  
            system_prompt = f"""You are a helpful workplace coach. NEVER mention frameworks or models.

Chat History:
{chat_history}

Current user message: "{user_message}"

🎯 TONE: Professional (like a trusted mentor)

📋 CRITICAL RULES:

**YOUR SCOPE:**
- ONLY help with workplace challenges: communication, conflicts, career growth, team dynamics, work stress
- If they ask about gossip, personal drama, or off-topic stuff → Redirect: "I'm here to assist with workplace challenges. What work-related matter can I help you with?"
- If it's an emergency (violence, harassment, mental health crisis) → Give crisis resources, don't try to coach

**CONVERSATION FLOW:**

**FIRST RESPONSE (when problem shared):**
- Acknowledge issue briefly (1 sentence)
- If you need 1-2 key details to help, ask ONE specific question
- If situation is clear, skip to solution immediately

**AFTER 1-2 CLARIFYING QUESTIONS:**
- STOP asking questions
- Give practical advice (1-2 sentences)
- If you lack specific company info (policies, procedures), acknowledge it: "I don't have access to your organization's specific policies, but typically you might..."
- End with simple yes/no question to keep engaged

**EXAMPLE - Leave request:**
User: "I want leave but used all my leave"
You: "That's certainly challenging. I don't have your organization's specific policies, but you might discuss options like unpaid leave or remote work with your supervisor if the need is urgent. Is this request time-sensitive?"

**EXAMPLE - Redirect gossip:**
User: "My coworker is dating the boss"
You: "I'm here to assist with workplace challenges. What work-related matter can I help you with?"

**EXAMPLE - Emergency redirect:**
User: "I want to hurt myself"
You: "⚠️ Please seek immediate support: National Suicide Prevention Lifeline: 988. I'm designed for workplace challenges, not crisis intervention."

**STYLE:**
- Max 2-3 sentences
- Professional but human: "I'd suggest...", "Consider...", "You might..."
- NO robotic phrases: "It sounds like...", "I understand that...", "Thank you for sharing..."

Respond in 2-3 sentences:"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,  # Higher for more natural/varied responses
            max_tokens=250  # Increased for complete 4-step responses
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
    
    # If GPT already added <br> tags, we're good - just clean up extras
    if '<br>' in text:
        # Clean up excessive line breaks (more than 2 in a row)
        text = re.sub(r'(<br>\s*){3,}', '<br><br>', text)
        return text
    
    # If no <br> tags, add them between bullets
    if '•' in text:
        # Split by bullet points
        lines = text.split('•')
        intro = lines[0].strip()
        bullets = []
        
        for line in lines[1:]:
            line = line.strip()
            if line:
                bullets.append('• ' + line)
        
        # Rejoin with proper line breaks
        if bullets:
            formatted_bullets = '<br>'.join(bullets)
            text = f"{intro}<br><br>{formatted_bullets}"
    
    # Replace numbered lists with bullets
    text = re.sub(r'(\d+)\.\s+\*\*([^*]+)\*\*', r'• <b>\2</b>', text)
    text = re.sub(r'(\d+)\.\s+<b>([^<]+)</b>', r'• <b>\2</b>', text)
    
    return text

# Initialize services on startup
initialize_services()

# Routes
@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle chat messages with JWT token-based sessions"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        original_user_message = user_message  # Save original before any modifications
        incoming_token = data.get('token', '')
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        logger.info(f"📨 User: {user_message}")
        logger.info(f"🔍 Token received: {'Yes' if incoming_token else 'No (new session)'}")
        
        # Decode existing token or create new session
        if incoming_token:
            session_data = decode_token(incoming_token)
            history = session_data['chat_history']
            selected_tone = session_data['tone']
            logger.info(f"✅ Decoded token - History length: {len(history)}, Tone: {selected_tone}")
        else:
            history = []
            selected_tone = None
            logger.info("✅ New session - No token provided")
        
        # Check if services are initialized
        if not qdrant_client or not openai_client:
            logger.error("Services not initialized, attempting to reinitialize...")
            if not initialize_services():
                return jsonify({'error': 'Services unavailable. Please try again later.', 'success': False}), 503
        
        # Check message limit (10 messages = 5 exchanges)
        current_count = len(history)
        if current_count >= 10:
            new_token = create_token(history, selected_tone)
            return jsonify({
                'response': "You've reached the free message limit (10 messages). Upgrade to Premium for unlimited conversations! 🚀",
                'limit_reached': True,
                'quick_replies': [],
                'token': new_token,
                'success': True
            })
        
        # Get relevant context from Qdrant using OpenAI embeddings
        context = get_relevant_context(user_message)
        
        # Build chat history string for context (last 4 exchanges)
        chat_history = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history[-4:]])
        
        # HANDLE TONE SELECTION
        if user_message.strip() in ["Professional", "Casual"]:
            selected_tone = user_message.strip()
            logger.info(f"✅ Tone '{selected_tone}' selected")
            
            # Get the user's problem from chat history
            user_messages = []
            for h in history:
                msg = h['user']
                # Skip greetings and tone selections
                if msg.lower() not in ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo', 'professional', 'casual', 'before i help you with this, how would you like me to respond?']:
                    user_messages.append(msg)
            
            # If we found their problem, REPLACE user_message with it
            if user_messages:
                user_message = user_messages[-1]  # Get the most recent problem statement
                logger.info(f"🔄 Responding to original problem: {user_message[:50]}...")
        
        # Check if this is a greeting (not a real problem)
        greeting_words = ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo', 'howdy']
        is_greeting = user_message.lower().strip() in greeting_words
        
        # Check if message is meaningful (not just 1-2 random words)
        word_count = len(user_message.split())
        is_meaningful_query = word_count >= 3  # Any message with 3+ words is considered real
        
        # If no tone selected and this is a real problem (not greeting), ask for tone FIRST
        if selected_tone is None and not is_greeting and is_meaningful_query:
            ai_response = "Before I help you with this, how would you like me to respond?"
            
            # Add to history
            history.append({
                'user': original_user_message,
                'ai': ai_response,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Create new token with updated history
            new_token = create_token(history, selected_tone)
            
            return jsonify({
                'response': ai_response,
                'quick_replies': ["Professional", "Casual"],
                'token': new_token,
                'success': True
            })
        
        # If it's just 1-2 random words (not meaningful), ask them to elaborate
        if selected_tone is None and not is_greeting and not is_meaningful_query:
            ai_response = "Could you tell me a bit more about what's going on?"
            
            # Add to history
            history.append({
                'user': original_user_message,
                'ai': ai_response,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Create new token
            new_token = create_token(history, selected_tone)
            
            return jsonify({
                'response': ai_response,
                'quick_replies': [],
                'token': new_token,
                'success': True
            })
        
        # Calculate chat length BEFORE adding current message
        current_chat_length = len(history) + 1
        
        # Generate response using GPT-4o-mini with Qdrant context
        ai_response = generate_response(user_message, context, chat_history, selected_tone, current_chat_length)
        
        # Add to history (use ORIGINAL message if tone was selected)
        history.append({
            'user': original_user_message,
            'ai': ai_response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        logger.info(f"✅ AI: {ai_response[:100]}...")
        
        # Smart quick reply flow
        is_safety_warning = ai_response.startswith("⚠️") or "call 911" in ai_response.lower()
        
        quick_replies = []
        
        # Never show buttons after safety warnings
        if is_safety_warning:
            quick_replies = []
            logger.info("⚠️ Safety warning - no buttons")
        
        # Create new token with updated history and tone
        new_token = create_token(history, selected_tone)
        
        response_data = jsonify({
            'response': ai_response,
            'quick_replies': quick_replies,
            'token': new_token,
            'success': True
        })
        
        # CORS headers
        response_data.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response_data.headers['Access-Control-Allow-Credentials'] = 'true'
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your message.',
            'success': False
        }), 500

@app.route('/api/history', methods=['GET', 'OPTIONS'])
def get_history():
    """Get chat history from JWT token"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response
    
    # Get token from Authorization header or query param
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        token = request.args.get('token', '')
    
    if not token:
        logger.info("🔍 No token provided - returning empty history")
        return jsonify({'history': []})
    
    # Decode token
    session_data = decode_token(token)
    history = session_data['chat_history']
    
    logger.info(f"🔍 History request - Token valid: {session_data.get('valid', False)}, History length: {len(history)}")
    
    response_data = jsonify({'history': history})
    
    # CORS headers
    response_data.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
    response_data.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response_data

@app.route('/api/clear', methods=['POST', 'OPTIONS'])
def clear_history():
    """Clear chat history - return new empty token"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response
    
    # Create new empty token
    new_token = create_token([], None)
    
    response_data = jsonify({
        'success': True,
        'message': 'Chat history cleared',
        'token': new_token
    })
    response_data.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
    response_data.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response_data

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

@app.route('/api/session-check')
def session_check():
    """Debug endpoint to check JWT token status"""
    # Get token from header or query
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        token = request.args.get('token', '')
    
    if not token:
        return jsonify({
            'has_token': False,
            'message': 'No token provided',
            'jwt_config': {
                'algorithm': JWT_ALGORITHM,
                'expiration_hours': JWT_EXPIRATION_HOURS
            }
        })
    
    # Decode token
    session_data = decode_token(token)
    
    return jsonify({
        'has_token': True,
        'token_valid': session_data.get('valid', False),
        'has_chat_history': len(session_data['chat_history']) > 0,
        'chat_length': len(session_data['chat_history']),
        'has_tone': session_data['tone'] is not None,
        'tone': session_data['tone'],
        'error': session_data.get('error'),
        'jwt_config': {
            'algorithm': JWT_ALGORITHM,
            'expiration_hours': JWT_EXPIRATION_HOURS
        }
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
