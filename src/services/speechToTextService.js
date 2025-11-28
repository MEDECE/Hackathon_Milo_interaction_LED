import axios from 'axios';

export const convertSpeechToText = async (audioBlob) => {
    try {
        const formData = new FormData();
        formData.append('file', audioBlob);

        const response = await axios.post('http://localhost:5000/api/speech-to-text', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            }
        });

        console.log('response', response.ok, response.status, response.data);

        if (response.status !== 200) {
            throw new Error('Speech to text conversion failed');
        }

        const data = await response.data;
        return data.text;
    } catch (error) {
        console.error('Speech to text service error:', error);
        throw error;
    }
};