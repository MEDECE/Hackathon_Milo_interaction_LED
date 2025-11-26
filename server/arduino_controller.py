"""
Contrôleur Arduino pour les LEDs de cohérence
Envoie le score de cohérence à l'Arduino via port série
"""

import serial
import serial.tools.list_ports
import time

class ArduinoController:
    def __init__(self, port=None, baudrate=9600):
        self.serial = None
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        
    def find_arduino_port(self):
        """Trouve automatiquement le port de l'Arduino"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Arduino apparaît souvent avec ces descriptions
            if 'Arduino' in port.description or 'CH340' in port.description or 'USB' in port.description:
                print(f"Arduino trouvé sur {port.device}")
                return port.device
        return None
    
    def connect(self):
        """Se connecte à l'Arduino"""
        try:
            if self.port is None:
                self.port = self.find_arduino_port()
            
            if self.port is None:
                print("Aucun Arduino trouvé. Mode simulation activé.")
                return False
            
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Attendre que l'Arduino se réinitialise
            self.connected = True
            print(f"Connecté à l'Arduino sur {self.port}")
            return True
        except Exception as e:
            print(f"Erreur de connexion à l'Arduino: {e}")
            self.connected = False
            return False
    
    def send_score(self, score: int):
        """Envoie le score de cohérence à l'Arduino"""
        score = max(0, min(100, score))  # Clamp entre 0 et 100
        
        if self.connected and self.serial:
            try:
                message = f"{score}\n"
                self.serial.write(message.encode())
                print(f"Score {score} envoyé à l'Arduino")
                
                # Lire la réponse de l'Arduino
                time.sleep(0.1)
                if self.serial.in_waiting:
                    response = self.serial.readline().decode().strip()
                    print(f"Réponse Arduino: {response}")
                return True
            except Exception as e:
                print(f"Erreur d'envoi à l'Arduino: {e}")
                return False
        else:
            # Mode simulation
            if score >= 70:
                print(f"[SIMULATION] Score {score}: 🟢 LED VERTE")
            elif score >= 40:
                print(f"[SIMULATION] Score {score}: 🟡 LED JAUNE")
            else:
                print(f"[SIMULATION] Score {score}: 🔴 LED ROUGE")
            return True
    
    def disconnect(self):
        """Déconnecte l'Arduino"""
        if self.serial:
            self.serial.close()
            self.connected = False
            print("Arduino déconnecté")


# Instance globale (singleton)
arduino = ArduinoController()


def init_arduino(port=None):
    """Initialise la connexion Arduino"""
    if port:
        arduino.port = port
    return arduino.connect()


def send_coherence_score(score: int):
    """Fonction simple pour envoyer un score"""
    return arduino.send_score(score)


# Test si exécuté directement
if __name__ == "__main__":
    print("Test du contrôleur Arduino")
    init_arduino()
    
    # Test des différents niveaux
    print("\nTest LED VERTE (score 85):")
    send_coherence_score(85)
    time.sleep(2)
    
    print("\nTest LED JAUNE (score 55):")
    send_coherence_score(55)
    time.sleep(2)
    
    print("\nTest LED ROUGE (score 20):")
    send_coherence_score(20)
    time.sleep(2)
    
    arduino.disconnect()
