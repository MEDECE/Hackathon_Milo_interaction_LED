export const textToSpeech = async (text) => {
    try {
        const response = await fetch('http://localhost:5000/api/text-to-speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        const data = await response.json();
        return data.success;
    } catch (error) {
        console.error('OpenAI API error:', error);
        throw error;
    }
};