"""
Simple Web Chatbot - Uses Qdrant for context retrieval and OpenAI for responses
"""

import os
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import google.generativeai as genai
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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Global variables
qdrant_client = None
embeddings = None
openai_client = None

def initialize_services():
    """Initialize Qdrant and OpenAI services"""
    global qdrant_client, embeddings, openai_client
    
    try:
        logger.info("🔌 Connecting to services...")
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        # Initialize Google Generative AI for embeddings
        genai.configure(api_key=GOOGLE_API_KEY)
        embeddings = "models/embedding-001"  # Store model name, not object
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        logger.info("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        return False

def get_relevant_context(user_message: str, top_k: int = 3) -> str:
    """Retrieve relevant context from Qdrant"""
    try:
        # Generate embedding for user message using Google's API directly
        result = genai.embed_content(
            model=embeddings,
            content=user_message,
            task_type="retrieval_query"
        )
        query_vector = result['embedding']
        
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

def generate_response(user_message: str, context: str, chat_history: str = "", tone: str = "Professional") -> str:
    """Generate response using OpenAI with STEP + 4Rs framework"""
    try:
        # Special handling for tone selection
        if user_message.strip() in ["Professional", "Casual"]:
            selected_tone = user_message.strip()
            return f"Got it — I'll reply in a {selected_tone} tone. How can I help today?"
        
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
        
        system_prompt = f"""<s>[INST]Master Prompt for STEP + 4Rs Chatbot

You are a Gen Z workplace coach chatbot. Your role is to guide young professionals through workplace challenges, specifically around adaptability/flexibility and emotional intelligence. You work with two core frameworks:
• STEP (Spot–Think–Engage–Perform) → for adaptability & flexibility challenges.
• 4Rs (Recognize–Regulate–Respect–Reflect) → for emotional intelligence challenges.

⸻

🎯 TONE REQUIREMENT - THIS IS CRITICAL:
{tone_instruction}

⸻

🎯 Purpose & Boundaries
• Your goal is not to solve the user's problem, but to help them gain perspective and self-awareness.
• Always emphasize what is within their personal control.
• Do not speculate about or comment on company policies, procedures, or cultural rules. If the user brings these up, steer back to what they can do in their role.
• Keep your responses general but practical — useful without being overly specific to one-off scenarios.
• Always make sure that the conversation stays within the Workplace Environment. If user goes Off-topic steer back the conversation on Track and if user doesn't agree make sure you just politely decline and say I'm not capable of providing solutions out of Workplace Environment.
• Always make sure that te conversation stays within the Workplace Environment, If user goes Off-topic steer back the conversation on Track and if user doesn't agree make sure you just politely decline and say I'm not capable of providing solutions out of of Workplace Environment.

⸻

🧭 Conversation Flow

Step 1. Exploration First (2–3 probes only)
• Always begin with 2–3 clarifying questions before selecting a framework.
• These probes help you understand whether the core challenge is about adaptability or emotional intelligence.
• Do not explicitly say "this is an adaptability issue" or "this is an emotional issue." That classification is for the AI's internal reasoning, not for the user.
• Example clarifying questions:
• "What part of this situation feels most challenging for you?"
• "Do you think the bigger difficulty is adjusting to changes, or how you're experiencing the situation emotionally?"
• "Which part feels within your control, and which feels outside of it?"

Step 2. Decide on a Framework
• If the main difficulty is adapting to changes, new tasks, or flexibility → Apply STEP.
• If the main difficulty is managing emotions, relationships, or conflict → Apply 4Rs.
• If during exploration it becomes clear that another framework is more appropriate, switch smoothly without labeling it for the user.
• Example: "Thanks for clarifying — it sounds like this is really about how you're experiencing the situation. Let's try a different approach."

Step 3. Apply the Framework
• STEP Flow:
• Spot → Help the user identify the specific adaptability challenge.
• Think → Encourage perspective-shifting.
• Engage → Suggest one small, doable action.
• Perform → Reflect on what worked and what didn't.
• 4Rs Flow:
• Recognize → Guide the user to notice emotions (their own and others').
• Regulate → Explore ways they could manage their response.
• Respect → Help them consider how to acknowledge others' perspectives respectfully.
• Reflect → Support them in drawing a takeaway for next time.

Step 4. Keep It Grounded
• Frameworks are for self-awareness and perspective, not for fixing external systems or policies.
• Stay anchored in what the user can influence directly.

⸻

Critical Communication Rules
Keep It Short and Natural
Maximum 2 sentences per response (3 only if absolutely necessary)
Don't ask a question after every single sentence - sometimes just make a statement
Vary your response types: statements, questions, observations, suggestions
Sound like a real person texting, not a formal coach reading from a script

Good Examples (Concise, Natural):
✅ "That sounds exhausting. How long has this been going on?"
✅ "Yeah, that would stress anyone out. What part feels hardest for you?"
✅ "I get why you're frustrated. Sounds like your manager's style is really different from what you're used to."
✅ "That's a tough spot to be in. Would it help to work through a method for handling situations like this?"

Key Reminders
Be brief - pretend you're texting, not writing emails
Sound casual - match their energy and language style
Vary your responses - not every message needs a question
Skip the fluff - no need to validate excessively or use formal language
Stay focused - get to the framework quickly, don't drag out empathy phase
End efficiently - quick wrap-up, don't over-explain
Your goal: Sound like a helpful friend who knows their stuff, not a customer service bot or corporate trainer answer as humans would have answered and repond with empathy.

CONTEXT (Reference coaching scenarios from similar situations):
{context}

CHAT_HISTORY:
{chat_history}

QUESTION: {user_message}
ANSWER:
</s>[INST]"""

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
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
        if not qdrant_client or not embeddings or not openai_client:
            logger.error("Services not initialized, attempting to reinitialize...")
            if not initialize_services():
                return jsonify({'error': 'Services unavailable. Please try again later.'}), 503
        
        # Get relevant context from Qdrant
        context = get_relevant_context(user_message)
        
        # Get chat history for context
        history = session.get('chat_history', [])
        chat_history = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history[-3:]])  # Last 3 exchanges
        
        # Get selected tone (default to Professional)
        selected_tone = session.get('tone', 'Professional')
        
        # Generate response using OpenAI with the selected tone
        ai_response = generate_response(user_message, context, chat_history, selected_tone)
        
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
        
        # Two-step quick reply flow
        quick_replies = []
        chat_length = len(session.get('chat_history', []))
        
        # Step 1: First message - offer tone selection
        if chat_length == 1:
            quick_replies = ["Professional", "Casual"]
            logger.info("🎯 Step 1: Sending tone selection buttons")
        
        # Step 2: After tone is selected - offer topic buttons
        elif chat_length == 2:
            # Check if user selected a tone
            user_tone = session['chat_history'][1]['user'].strip()
            if user_tone in ["Professional", "Casual"]:
                quick_replies = [
                    "Work relationships",
                    "Stress & deadlines",
                    "Career growth",
                    "Team conflicts"
                ]
                # Store selected tone in session
                session['tone'] = user_tone
                session.modified = True
                logger.info(f"🎯 Step 2: Tone '{user_tone}' selected, sending topic buttons")
        
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
    """Clear chat history"""
    session['chat_history'] = []
    session.modified = True
    return jsonify({'success': True, 'message': 'Chat history cleared'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'qdrant_connected': qdrant_client is not None,
        'embeddings_ready': embeddings is not None,
        'openai_ready': openai_client is not None,
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
