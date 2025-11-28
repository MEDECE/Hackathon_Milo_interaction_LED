import React from 'react';
import botAvatar from '../assets/bot-avatar.png';
import userAvatar from '../assets/user-avatar.png';

// Retourne la couleur du badge selon le score de cohérence
const getCoherenceColor = (score) => {
  if (score >= 70) return '#22c55e'; // Vert
  if (score >= 40) return '#eab308'; // Jaune
  return '#ef4444'; // Rouge
};

const getCoherenceLabel = (score) => {
  if (score >= 70) return 'Cohérent';
  if (score >= 40) return 'Moyen';
  return 'Incohérent';
};

const ChatMessage = ({ message }) => {
  const isBot = message.role === 'assistant';
  const coherence = message.coherence;

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        marginBottom: '20px',
        flexDirection: isBot ? 'row' : 'row-reverse'
      }}
    >
      <div
        style={{
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          marginRight: isBot ? '12px' : '0',
          marginLeft: isBot ? '0' : '12px',
          overflow: 'hidden'
        }}
      >
        <img
          src={isBot ? botAvatar : userAvatar}
          alt={isBot ? 'Milo (IA)' : 'User'}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
      </div>
      <div
        style={{
          background: isBot ? '#f0f0f0' : '#007178',
          color: isBot ? '#333' : 'white',
          padding: '12px 16px',
          borderRadius: '12px',
          maxWidth: '70%'
        }}
      >
        <div style={{ fontWeight: '600', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isBot ? 'Milo (IA)' : 'Vous'}
          {/* Badge de cohérence pour les réponses de l'assistant */}
          {isBot && coherence !== undefined && (
            <span
              style={{
                backgroundColor: getCoherenceColor(coherence),
                color: 'white',
                padding: '2px 8px',
                borderRadius: '12px',
                fontSize: '11px',
                fontWeight: '500'
              }}
            >
              {coherence}% - {getCoherenceLabel(coherence)}
            </span>
          )}
        </div>
        <div style={{ whiteSpace: 'pre-wrap' }}>{message.content}</div>
      </div>
    </div>
  );
};

export default ChatMessage;