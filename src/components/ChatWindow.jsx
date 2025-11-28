import React, { forwardRef, useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';

const ChatWindow = forwardRef(({ messages, isLoading }, ref) => {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div
            ref={ref}
            style={{
                background: 'white',
                borderRadius: '12px',
                height: '600px',
                overflowY: 'auto',
                padding: '20px',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
            }}
        >
            {messages.map((message, index) => {
                if (message?.role === 'system') return null
                return (
                    <ChatMessage key={index} message={message} />
                )
            })}
            {isLoading && (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                    <div
                        style={{
                            display: 'inline-block',
                            width: '20px',
                            height: '20px',
                            border: '2px solid #007178',
                            borderTopColor: 'transparent',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite',
                        }}
                    />
                </div>
            )}
            <div ref={bottomRef} />
        </div>
    );
});

export default ChatWindow;