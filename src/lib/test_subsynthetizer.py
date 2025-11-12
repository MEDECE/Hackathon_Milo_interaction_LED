from subsynthetizer import SubSynthesizer

synthetizer = SubSynthesizer()

question = "Comment t'appelles-tu ?"
response, coherence = synthetizer.analyze_prompt(question)

print("\n✅ Réponse :", response)
print("📊 Score de cohérence :", coherence, "%")

if coherence < 50 :
    print("🔴 LED Rouge : La réponse est peu cohérente.")
elif coherence < 80 :
    print("🟡 LED Jaune : La réponse est moyennement cohérente.")
else :
    print("🟢 LED Verte : La réponse est très cohérente.")