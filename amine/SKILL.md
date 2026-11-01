---
name: amine
description: Création des vidéos d'Amine (voix ElevenLabs, texte Groq, images/vidéos Replicate, montage ffmpeg).
---

# Vidéos Amine

Outils disponibles dans l'environnement :
- `ffmpeg` / `ffprobe` pour le montage
- `yt-dlp` pour télécharger des vidéos
- SDK Python `elevenlabs`, `groq`, `replicate` (les clés API sont injectées automatiquement)

Les dépendances sont installées au démarrage par `setup.sh` à la racine du dépôt.
