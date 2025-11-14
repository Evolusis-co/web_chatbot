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

def generate_response(user_message: str, context: str, chat_history: str = "", tone: str = None, chat_length: int = 0) -> str:
    """Generate response using GPT-4o-mini with STEP + 4Rs framework and Qdrant context"""
    try:
        # Check if this is a greeting (first message ONLY)
        greeting_words = ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo', 'helo', 'hola']
        if user_message.lower().strip() in greeting_words and chat_length == 0:
            # Return friendly, natural greeting (no tone needed for greetings)
            return "Hello! I'm here to help with any challenges you're facing. What's on your mind?"
        
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
            system_prompt = f"""You are an AI coach trained on the STEP Framework and the 4R Emotional Professionalism Framework.
These frameworks guide your thinking, but you must NEVER reveal their names or steps.

User's issue: "{user_message}"

🎯 TONE: Casual (friendly, approachable colleague)

📋 CRITICAL RULES:
1. Give DIRECT, ACTIONABLE advice - NOT therapy questions
2. If user asks about leave/time off → Tell them HOW to request it
3. If user has a problem → Tell them WHAT to do about it
4. Keep it SHORT (2-4 sentences max)
5. Write like natural conversation - NO bullet points, NO numbered lists
6. NEVER mention "STEP", "4Rs", or any framework names
7. NEVER ask "What's making you feel...?" or "Why do you...?" - Give solutions instead
8. Use contractions: "you're", "don't", "can't"
9. Help with ANY employee problem - workplace or personal

Example for leave request:
"Check your company's leave policy first, then send a quick email to your manager. Keep it simple - just say you need a day off and when. If it's urgent, follow up with a call."

Example for conflict:
"Try talking to them directly in private. Be calm, explain how their actions affect you, and ask if they can adjust. Most people respond well to honest, respectful feedback."

Give practical, specific advice (40-60 words):"""
        else:  # Professional  
            system_prompt = f"""You are an AI coach trained on the STEP Framework and the 4R Emotional Professionalism Framework.
These frameworks guide your thinking, but you must NEVER reveal their names or steps.

User's issue: "{user_message}"

🎯 TONE: Professional (polished, respectful mentor)

📋 CRITICAL RULES:
1. Give DIRECT, ACTIONABLE advice - NOT therapy questions
2. If user asks about leave/time off → Tell them HOW to request it
3. If user has a problem → Tell them WHAT to do about it
4. Keep it SHORT (2-4 sentences max)
5. Write like natural conversation - NO bullet points, NO numbered lists
6. NEVER mention "STEP", "4Rs", or any framework names
7. NEVER ask "What's making you feel...?" or "Why do you...?" - Give solutions instead
8. Use formal language: "I recommend", "Consider", "It would be beneficial"
9. Help with ANY employee problem - workplace or personal

Example for leave request:
"Review your organization's leave policy to understand the process, then submit a formal request to your supervisor with the dates and reason. Ensure you provide adequate notice and arrange coverage for your responsibilities."

Example for conflict:
"I recommend scheduling a private conversation to address this professionally. Clearly communicate your concerns using specific examples, and propose collaborative solutions. Documenting the discussion may also be beneficial."

Give practical, specific advice (40-60 words):"""

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
        
        # HANDLE TONE SELECTION FIRST
        if user_message.strip() in ["Professional", "Casual"]:
            selected_tone = user_message.strip()
            
            # Save the selected tone
            session['tone'] = selected_tone
            session.modified = True
            logger.info(f"✅ Tone '{selected_tone}' saved to session")
            
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
                logger.info(f"🔄 Tone selected: {selected_tone}. Responding to original problem: {user_message[:50]}...")
        
        # Get selected tone (NO DEFAULT - should be None if not set)
        selected_tone = session.get('tone', None)
        
        # Check if this is a greeting (not a real problem)
        greeting_words = ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo', 'howdy']
        is_greeting = user_message.lower().strip() in greeting_words
        
        # Check if message is meaningful (not just 1-2 random words)
        word_count = len(user_message.split())
        is_meaningful_query = word_count >= 3  # Any message with 3+ words is considered real
        
        # If no tone selected and this is a real problem (not greeting), ask for tone FIRST
        if selected_tone is None and not is_greeting and is_meaningful_query:
            ai_response = "Before I help you with this, how would you like me to respond?"
            
            # Store in history
            if 'chat_history' not in session:
                session['chat_history'] = []
            
            session['chat_history'].append({
                'user': original_user_message,
                'ai': ai_response,
                'timestamp': datetime.now().isoformat()
            })
            session.modified = True
            
            return jsonify({
                'response': ai_response,
                'quick_replies': ["Professional", "Casual"],
                'success': True
            })
        
        # If it's just 1-2 random words (not meaningful), ask them to elaborate
        if selected_tone is None and not is_greeting and not is_meaningful_query:
            ai_response = "Could you tell me a bit more about what's going on?"
            
            if 'chat_history' not in session:
                session['chat_history'] = []
            
            session['chat_history'].append({
                'user': original_user_message,
                'ai': ai_response,
                'timestamp': datetime.now().isoformat()
            })
            session.modified = True
            
            return jsonify({
                'response': ai_response,
                'quick_replies': [],
                'success': True
            })
        
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
        
        # Store tone if user just selected it (check ORIGINAL message)
        if original_user_message.strip() in ["Professional", "Casual"]:
            session['tone'] = original_user_message.strip()
            session.modified = True
            logger.info(f"✅ Tone '{original_user_message.strip()}' saved to session")
        
        # Never show buttons after safety warnings or if tone already selected
        if is_safety_warning:
            quick_replies = []
            logger.info("⚠️ Safety warning - no buttons")
        
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
