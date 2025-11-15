# Frontend Changes Required for JWT Token-Based Sessions

## Overview
The backend now uses **JWT tokens** instead of cookies for session management. This fixes Safari/iOS/Mac session issues and scales better.

## What Changed
- ❌ **Old**: Flask sessions stored on server, sent via cookies
- ✅ **New**: JWT tokens containing session data, sent in request body

## Required Frontend Changes

### 1. Update `httpService.js`

**Before:**
```js
export const sendMessage = async (message) => {
  if (!message || !message.trim()) {
    throw new Error("Message cannot be empty");
  }
  return chatbotApi.post("/api/chat", { message });
};
```

**After:**
```js
export const sendMessage = async (message, token = null) => {
  if (!message || !message.trim()) {
    throw new Error("Message cannot be empty");
  }
  return chatbotApi.post("/api/chat", { 
    message,
    token  // Send token in request body
  });
};

export const getHistory = async (token = null) => {
  // Send token as query parameter
  const params = token ? { token } : {};
  return chatbotApi.get("/api/history", { params });
};

export const clearHistory = async () => {
  return chatbotApi.post("/api/clear");
};
```

---

### 2. Update `useChatbot.js` Hook

**Key changes:**
- Store token in `localStorage` (persists after refresh)
- Send token with every request
- Update token when backend returns new one

```js
// src/services/chatbot/useChatBot.js
import { useState, useEffect } from "react";
import { sendMessage, getHistory, clearHistory, checkHealth } from "./httpService";

const TOKEN_KEY = 'chatbot_token';  // LocalStorage key

export function useChatbot() {
  const [messages, setMessages] = useState([]);
  const [quickReplies, setQuickReplies] = useState([]);
  const [limitReached, setLimitReached] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isHealthy, setIsHealthy] = useState(null);
  const [error, setError] = useState(null);
  const [token, setToken] = useState(() => {
    // Load token from localStorage on init
    return localStorage.getItem(TOKEN_KEY) || null;
  });

  // Save token to localStorage whenever it changes
  useEffect(() => {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_KEY);
    }
  }, [token]);

  // --- Health check on mount ---
  useEffect(() => {
    (async () => {
      try {
        const res = await checkHealth();
        setIsHealthy(res.status === "healthy");
      } catch (err) {
        console.error("Health check failed:", err);
        setIsHealthy(false);
      }
    })();
  }, []);

  // --- Load session history from token ---
  const loadHistory = async () => {
    try {
      const data = await getHistory(token);
      const restored = [];
      for (const entry of data.history || []) {
        restored.push({ text: entry.user, role: "user" });
        restored.push({ text: entry.ai, role: "ai" });
      }
      setMessages(restored);
    } catch (err) {
      setError(err);
    }
  };

  // --- Clear conversation ---
  const clearChat = async () => {
    try {
      const data = await clearHistory();
      setMessages([]);
      setQuickReplies([]);
      setLimitReached(false);
      // Update token with new empty one from backend
      if (data.token) {
        setToken(data.token);
      } else {
        setToken(null);
      }
    } catch (err) {
      setError(err);
    }
  };

  // --- Send message ---
  const sendChatMessage = async (text) => {
    if (!text?.trim()) return;
    setMessages((prev) => [...prev, { text, role: "user" }]);
    setIsLoading(true);
    setError(null);

    try {
      // Send token with message
      const data = await sendMessage(text, token);

      if (data.success) {
        setMessages((prev) => [...prev, { text: data.response, role: "ai" }]);
        setQuickReplies(data.quick_replies || []);
        setLimitReached(Boolean(data.limit_reached));
        
        // **CRITICAL**: Update token with new one from backend
        if (data.token) {
          setToken(data.token);
        }
      } else {
        setError(data.error || "Unknown API error");
      }
    } catch (err) {
      setError(err);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    // State
    messages,
    quickReplies,
    limitReached,
    isLoading,
    isHealthy,
    error,
    token,  // Export token for debugging

    // Actions
    sendChatMessage,
    loadHistory,
    clearChat,
  };
}
```

---

### 3. Remove `withCredentials` (No Longer Needed)

**In `httpService.js`, remove this line:**
```js
chatbotApi.defaults.withCredentials = true;  // ❌ DELETE THIS
```

**Why?** We're not using cookies anymore, so credentials are unnecessary.

---

## Testing Checklist

✅ **1. First message** → Backend creates token, frontend stores it  
✅ **2. Second message** → Frontend sends token, backend returns updated token  
✅ **3. Refresh page** → Token loaded from localStorage, history restored  
✅ **4. Clear chat** → Token cleared, new empty token created  
✅ **5. Safari/iOS** → Works without cookie blocking issues  
✅ **6. Chrome/Edge** → Works consistently  
✅ **7. Message limit** → Token still returned with limit message  

---

## Debugging

### Check if token is saved:
```js
console.log('Token:', localStorage.getItem('chatbot_token'));
```

### Check token validity:
```
GET https://web-chatbot-qjuw.onrender.com/api/session-check?token=YOUR_TOKEN
```

### Backend logs will show:
```
🔍 Token received: Yes
✅ Decoded token - History length: 3, Tone: Casual
```

---

## Benefits of JWT Tokens

✅ **Safari/iOS compatibility** - No third-party cookie issues  
✅ **Scalability** - No server-side session storage  
✅ **Survives Render restarts** - Session data lives in token  
✅ **Works after page refresh** - Token persists in localStorage  
✅ **Simpler architecture** - No sticky sessions needed  

---

## Migration Notes

- **No breaking changes** for the UI component
- Only `httpService.js` and `useChatbot.js` need updates
- Token is automatically managed (no manual handling needed)
- **Old sessions will be lost** (users start fresh after deployment)

---

## Questions?

If you see errors:
1. Check browser console for token value
2. Check Network tab → Request payload includes `token` field
3. Check `/api/session-check` endpoint for token validity
4. Verify backend deployed successfully (commit e0cb4a6 + new changes)
