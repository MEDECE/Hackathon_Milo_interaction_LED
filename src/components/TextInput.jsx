import React, { useState } from 'react';
import SubmitButton from './SubmitButton';

const TextInput = ({ onSend, disabled }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input, 'text');
      setInput('');
    }
  };

  return (
    <form 
      onSubmit={handleSubmit}
      style={{ display: 'flex', flex: 1, gap: '10px', alignItems: 'center' }}
    >
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        disabled={disabled}
        placeholder="Écrivez votre message..."
        style={{
          flex: 1,
          padding: '12px 16px',
          borderRadius: '25px',
          border: '1px solid #ddd',
          fontSize: '16px',
          outline: 'none'
        }}
      />
      <SubmitButton
        disabled={disabled || !input.trim()}
        onClick={handleSubmit}
        />
      
    </form>
  );
};

export default TextInput;