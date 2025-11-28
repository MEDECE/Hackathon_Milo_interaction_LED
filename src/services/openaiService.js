export const sendMessageToOpenAI = async (messages) => {
    try {
        const response = await fetch('http://localhost:5000/api/ask-gpt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ messages }),
        });

        const data = await response.json();
        // Retourne la réponse et le score de cohérence
        return {
            response: data.response,
            coherence: data.coherence ?? 50  // Valeur par défaut si non fourni
        };
    } catch (error) {
        console.error('OpenAI API error:', error);
        throw error;
    }
};