import { FaTrash } from 'react-icons/fa';

function ClearButton({ setMessages, messages, isLoading }) {
  return (
    <button
      onClick={() => setMessages([messages[0], messages[1]])}
      disabled={isLoading}
      style={{
        background: 'white',
        color: 'red',  // La couleur de l'icône
        border: '2px solid white', // Optionnel : bordure rouge autour du bouton
        borderRadius: '50%',     // Forme ronde
        width: '40px',           // Taille du bouton
        height: '40px',          // Taille du bouton
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: isLoading ? 'not-allowed' : 'pointer',
        opacity: isLoading ? 0.7 : 1,
        padding: 0
      }}
    >
      <FaTrash size={17} />
      {/* Texte optionnel, supprime si non nécessaire */}
      {/* <span>Effacer</span> */}
    </button>
  );
}

export default ClearButton;