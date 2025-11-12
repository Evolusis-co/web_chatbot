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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
CORS(app)

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
        
        # Special handling for message 3 - Ask about tone preference WITH CLEAR EXPLANATION
        if chat_length == 3:
            return "I can help you with this. Quick question — would you prefer my advice in a **casual, friendly tone** or a **professional, formal tone**? Either way works, just pick what feels comfortable for you."
        
        # Special handling for tone selection - Use chat history to continue the conversation
        if user_message.strip() in ["Professional", "Casual"]:
            selected_tone = user_message.strip()
            
            # Get the user's last substantive message (not just tone selection)
            if chat_history:
                # Extract the last user message from chat history
                history_lines = chat_history.strip().split('\n')
                last_user_msg = None
                for line in reversed(history_lines):
                    if line.startswith('User:'):
                        last_user_msg = line.replace('User:', '').strip()
                        break
                
                # If we found their problem, respond to it directly in the selected tone
                if last_user_msg and last_user_msg not in ["Professional", "Casual"]:
                    # Don't just acknowledge tone - jump straight into helping with their problem
                    # The system prompt will handle the tone, just return empty to let GPT respond
                    pass  # Let it fall through to normal GPT response
            
            # If no context, just acknowledge
            return f"Perfect! I'll keep it {selected_tone.lower()}. What's the situation?"
        
        # Define tone-specific instructions
        if tone == "Casual":
            tone_instruction = """• Use a CASUAL, Gen Z tone: relaxed, conversational, like texting a smart friend
• Use phrases like: "That sucks", "Ugh that's annoying", "Yeah I get it", "Super frustrating"
• Use contractions: "you're", "that's", "don't", "can't"
• Keep it SHORT and NATURAL - sound like you're texting, not writing an essay
• Be supportive but chill: "Okay let's figure this out" instead of "I understand your concern"
• Example casual response: "Ugh that's super frustrating. So the main issue is they're ghosting you? What part bothers you most - them ignoring you or how it makes you look?"
"""
        else:  # Professional
            tone_instruction = """• Use a PROFESSIONAL tone: measured, empathetic, but formal like a workplace mentor or HR coach
• Use complete sentences with proper grammar
• Use phrases like: "I understand this is challenging", "That's a difficult situation", "Let's explore this together"
• Be empathetic but maintain professional distance
• Avoid slang or Gen Z casual language
• Example professional response: "That's a challenging situation. It sounds like communication barriers are impacting your work. Have you had an opportunity to address this directly with your colleague?"
"""
        
        system_prompt = f"""You are a Gen Z workplace coach helping young professionals with work challenges.

Use these frameworks:
• STEP (Adaptability): Spot → Think → Engage → Perform
• 4Rs (Emotional Intelligence): Recognize → Regulate → Respect → Reflect

⸻

🎯 CRITICAL FORMATTING INSTRUCTIONS:

Your response will be displayed in a web chat interface. Format it using HTML:

1. Use <b>keyword</b> for bold text (e.g., <b>Spot</b>, <b>Think</b>)
2. Use <br><br> for line breaks between bullet points
3. Use • symbol for bullets

Example format:
"Ugh that sounds rough. Here's how to handle it with STEP:<br><br>• <b>Spot</b> - Figure out which tasks are urgent<br><br>• <b>Think</b> - Your boss might not realize they're overloading you<br><br>• <b>Engage</b> - Ask for a quick 15-min priority check<br><br>• <b>Perform</b> - Track progress and escalate if needed<br><br>Sound good?"

KEY RULES:
- NEVER use **text** for bold, use <b>text</b> instead
- NEVER use \\n for line breaks, use <br><br> instead
- NEVER use numbered lists (1. 2. 3.), always use bullets (•)
- Keep total response under 120 words

⸻

🎯 TONE:
{tone_instruction}

⸻

🎯 CONVERSATION APPROACH:

**First response to a problem:** Ask 1 clarifying question to understand better
**Second response onwards:** Give actionable STEP or 4Rs advice with bullets

⸻

**CONTEXT FROM YOUR DATASET:**
{context}

**PREVIOUS CONVERSATION:**
{chat_history}

**USER JUST SAID:**
{user_message}

**YOUR RESPONSE (Use <b> for bold, <br><br> between bullets):**"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini for smarter responses
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,  # Slightly higher for more creative, natural responses
            max_tokens=200  # Increased to allow more detailed framework explanations
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Sorry, I'm having trouble generating a response right now. Please try again."

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
        
        # Store in session history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'user': user_message,
            'ai': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        session.modified = True
        
        logger.info(f"✅ AI: {ai_response[:100]}...")
        
        # Simplified quick reply flow
        # DON'T show buttons if the response is a safety warning
        is_safety_warning = ai_response.startswith("⚠️") or "call 911" in ai_response.lower()
        
        # Check if user has selected tone
        has_selected_tone = 'tone' in session and session['tone'] in ["Professional", "Casual"]
        
        quick_replies = []
        chat_length = len(session.get('chat_history', []))
        
        if is_safety_warning:
            # Never show buttons after safety warnings
            quick_replies = []
            logger.info("⚠️ Safety warning issued, no buttons shown")
        
        # Only show tone buttons at message 3 (after greeting + problem + bot's clarifying question)
        elif chat_length == 3 and not has_selected_tone:
            # Check if this is the normal flow: greeting → problem → clarifying question
            first_msg = session['chat_history'][0]['user'].lower().strip()
            greeting_words = ['hi', 'hello', 'hey', 'hii', 'hiii', 'sup', 'yo']
            
            if first_msg in greeting_words:
                # Check if the bot's previous response was asking a question (not giving advice)
                prev_ai_response = session['chat_history'][1]['ai']
                if '?' in prev_ai_response:  # Bot asked a clarifying question
                    quick_replies = ["Professional", "Casual"]
                    logger.info("🎯 Message 3: Showing tone buttons after clarifying question")
        
        # Store tone if user just selected it
        if user_message.strip() in ["Professional", "Casual"]:
            session['tone'] = user_message.strip()
            session.modified = True
            logger.info(f"✅ Tone '{user_message.strip()}' saved to session")
        
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
