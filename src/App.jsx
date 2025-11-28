import React, { useState, useRef } from 'react';
import ChatWindow from './components/ChatWindow';
import TextInput from './components/TextInput';
import AudioInput from './components/AudioInput';
import ClearButton from './components/ClearButton';
import { sendMessageToOpenAI } from './services/openaiService';
import { convertSpeechToText } from './services/speechToTextService';
import { textToSpeech } from './services/textToSpeechService';
import { get_instructions } from './services/getInstructions';
import schoolLogo from './assets/logo.png';

const App = () => {
  const [messages, setMessages] = useState([
    { "role": "system", "content": get_instructions() },
    { "role": "assistant", "content": "Bonjour et bienvenue à l'ECE Paris ! Comment puis-je vous aider aujourd'hui ? Je suis là pour répondre à toutes vos questions sur l'école." }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const chatWindowRef = useRef(null);

  const handleSendMessage = async (content, type = 'text') => {
    if (type === 'text') {
      if (!content.trim()) {
        return;
      }
    }

    setIsLoading(true);
    let messageText = content;

    // If it's an audio message, convert it to text first
    if (type === 'audio') {
      try {
        messageText = await convertSpeechToText(content);
      } catch (error) {
        console.error('Speech to text conversion failed:', error);
        setIsLoading(false);
        return;
      }
    }

    // Add user message
    const newMessages = [...messages, { role: 'user', content: messageText.trim() }];
    setMessages(newMessages);

    try {
      // Get bot response (now includes coherence score)
      const { response, coherence } = await sendMessageToOpenAI(newMessages);
      setMessages([...newMessages, { role: 'assistant', content: response, coherence: coherence }]);
      await textToSpeech(response);
    } catch (error) {
      console.error('Failed to get bot response:', error);
    } finally {
      console.log('finally')
      setIsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      <header style={{ textAlign: 'center', marginBottom: '20px' }}>
        <img
          src={schoolLogo}
          alt="School Logo"
          style={{ height: '90px', marginBottom: '10px' }}
        />
        {/*<h1 style={{ color: 'white', fontSize: '24px' }}>Assistant IA de l'École</h1>*/}
      </header>

      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        ref={chatWindowRef}
      />

      <div style={{ display: 'flex', gap: '10px', marginTop: '20px', alignItems: 'center' }}>

        <ClearButton
          setMessages={setMessages}
          messages={messages}
          isLoading={isLoading}
        />
        <TextInput
          onSend={handleSendMessage}
          disabled={isLoading}
        />
        {false && <AudioInput
          onRecordingComplete={handleSendMessage}
          disabled={isLoading}
        />}
      </div>
    </div>
  );
};

export default App;