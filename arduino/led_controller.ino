/*
 * Contrôleur de LEDs pour l'assistant vocal ECE
 * 
 * Branchements:
 * - LED Verte : Pin 3 (avec résistance 220Ω)
 * - LED Jaune : Pin 5 (avec résistance 220Ω)  
 * - LED Rouge : Pin 6 (avec résistance 220Ω)
 * - GND commun
 * 
 * Communication série à 9600 bauds
 * Envoie un nombre 0-100 pour définir l'état des LEDs
 */

const int LED_VERTE = 3;
const int LED_JAUNE = 5;
const int LED_ROUGE = 6;

void setup() {
  Serial.begin(9600);
  
  pinMode(LED_VERTE, OUTPUT);
  pinMode(LED_JAUNE, OUTPUT);
  pinMode(LED_ROUGE, OUTPUT);
  
  // Test au démarrage : allumer chaque LED brièvement
  testLeds();
  
  Serial.println("Arduino LED Controller pret!");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    int score = input.toInt();
    
    // S'assurer que le score est entre 0 et 100
    score = constrain(score, 0, 100);
    
    updateLeds(score);
    
    Serial.print("Score recu: ");
    Serial.println(score);
  }
}

void updateLeds(int score) {
  // Éteindre toutes les LEDs d'abord
  digitalWrite(LED_VERTE, LOW);
  digitalWrite(LED_JAUNE, LOW);
  digitalWrite(LED_ROUGE, LOW);
  
  if (score >= 70) {
    // Cohérent -> LED Verte
    digitalWrite(LED_VERTE, HIGH);
    Serial.println("LED VERTE allumee (coherent)");
  } 
  else if (score >= 40) {
    // Moyennement cohérent -> LED Jaune
    digitalWrite(LED_JAUNE, HIGH);
    Serial.println("LED JAUNE allumee (moyen)");
  } 
  else {
    // Incohérent -> LED Rouge
    digitalWrite(LED_ROUGE, HIGH);
    Serial.println("LED ROUGE allumee (incoherent)");
  }
}

void testLeds() {
  // Test séquentiel des LEDs au démarrage
  digitalWrite(LED_VERTE, HIGH);
  delay(300);
  digitalWrite(LED_VERTE, LOW);
  
  digitalWrite(LED_JAUNE, HIGH);
  delay(300);
  digitalWrite(LED_JAUNE, LOW);
  
  digitalWrite(LED_ROUGE, HIGH);
  delay(300);
  digitalWrite(LED_ROUGE, LOW);
}
