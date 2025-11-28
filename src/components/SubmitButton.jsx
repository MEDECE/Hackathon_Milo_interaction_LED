import { GrSend } from "react-icons/gr";

function ClearButton({ onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: 'white',
        color: 'black',  // La couleur de l'icône
        border: '2px solid white', // Optionnel : bordure rouge autour du bouton
        borderRadius: '50%',     // Forme ronde
        width: '40px',           // Taille du bouton
        height: '40px',          // Taille du bouton
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.7 : 1,
        padding: 0
      }}
    >
      <GrSend size={20}/>
      {/* Texte optionnel, supprime si non nécessaire */}
      {/* <span>Effacer</span> */}
    </button>
  );
}

export default ClearButton;