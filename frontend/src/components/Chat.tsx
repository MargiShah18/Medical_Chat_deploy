import { useState, useEffect, useRef } from "react";
import { ChatMessage as ChatMessageType } from "../types";
import ChatMessage from "./ChatMessage";

const Chat = () => {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const synthesisRef = useRef<SpeechSynthesis | null>(null);

  useEffect(() => {
    // Initialize speech recognition
    if ('webkitSpeechRecognition' in window) {
      recognitionRef.current = new (window.webkitSpeechRecognition as any)();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
        // Automatically send the message when speech recognition is complete
        if (transcript.trim()) {
          handleSubmit(new Event('submit') as any, transcript);
        }
      };

      recognitionRef.current.onerror = (event: SpeechRecognitionError) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
    }

    // Initialize speech synthesis
    synthesisRef.current = window.speechSynthesis;
  }, []);

  const startListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const stopSpeaking = () => {
    if (synthesisRef.current) {
      synthesisRef.current.cancel();
      setIsSpeaking(false);
    }
  };

  const speakText = (text: string) => {
    // Stop any ongoing speech recognition when starting to speak
    stopListening();
    
    if (synthesisRef.current) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      synthesisRef.current.speak(utterance);
    }
  };

  const handleSubmit = async (e: React.FormEvent, transcript?: string) => {
    e.preventDefault();
    const messageText = transcript || input;
    if (!messageText.trim()) return;

    const userMessage: ChatMessageType = {
      role: "user",
      content: messageText,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:5001/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        mode: "cors",
        cache: "no-cache",
        body: JSON.stringify({ query: userMessage.content }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: ChatMessageType = {
        role: "assistant",
        content: data.response,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      // Speak the assistant's response
      speakText(data.response);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: ChatMessageType = {
        role: "assistant",
        content: "Sorry, there was an error processing your request.",
      };
      setMessages((prev) => [...prev, errorMessage]);
      speakText(errorMessage.content);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 max-w-3xl">
      <div className="flex flex-col h-[600px] border border-gray-200 rounded-xl shadow-lg bg-white overflow-hidden">
        <div className="p-4 border-b border-gray-200 bg-white">
          <h2 className="text-lg font-semibold text-gray-800">
            Chat with Your PDF
          </h2>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
          {messages.map((message, index) => (
            <ChatMessage key={index} message={message} />
          ))}
          {loading && (
            <div className="flex justify-center items-center py-4">
              <div className="dot-flashing"></div>
            </div>
          )}
        </div>
        <form
          onSubmit={handleSubmit}
          className="p-4 border-t border-gray-200 bg-white"
        >
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your document..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg
                bg-white text-gray-900 
                focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                disabled:bg-gray-100"
              disabled={loading || isSpeaking}
            />
            <button
              type="button"
              onClick={isListening ? stopListening : startListening}
              disabled={loading || isSpeaking}
              className={`px-4 py-3 rounded-lg transition duration-200 ease-in-out
                font-medium shadow-sm whitespace-nowrap
                ${isListening 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-green-600 hover:bg-green-700 text-white'}`}
            >
              {isListening ? 'Stop' : '🎤'}
            </button>
            {isSpeaking && (
              <button
                type="button"
                onClick={stopSpeaking}
                className="px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 
                  transition duration-200 ease-in-out
                  font-medium shadow-sm whitespace-nowrap"
              >
                🔊
              </button>
            )}
            <button
              type="submit"
              disabled={loading || isSpeaking}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                disabled:bg-blue-300 disabled:cursor-not-allowed
                transition duration-200 ease-in-out
                font-medium shadow-sm whitespace-nowrap"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Chat;
