"""Clients prêts à l'emploi pour les vidéos d'Amine.

Les vraies clés API sont ajoutées par le proxy de l'environnement : on passe des
valeurs factices. Le SDK Replicate crée son propre transport httpx, qui ignore
HTTPS_PROXY ; on lui en fournit un explicitement, sinon les appels échouent en 401.

Usage :
    import sys; sys.path.insert(0, ".claude/skills/amine")
    from clients import elevenlabs, groq, replicate
"""
import os

import httpx
import replicate as _replicate
from elevenlabs.client import ElevenLabs
from groq import Groq

_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

elevenlabs = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", "dummy"))
groq = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy"))
replicate = _replicate.Client(
    api_token=os.environ.get("REPLICATE_API_TOKEN", "dummy"),
    **({"transport": httpx.HTTPTransport(proxy=_proxy)} if _proxy else {}),
)

# Les modèles "officiels" (ex. black-forest-labs/flux-schnell) renvoient "version": null,
# que le SDK 1.0.7 refuse (erreur "none is not an allowed value"). On rend le champ optionnel.
from replicate.prediction import Prediction as _Prediction

_field = _Prediction.__fields__["version"]
_field.required = False
_field.allow_none = True

GROQ_MODEL = "openai/gpt-oss-120b"


def telecharger(sortie, chemin):
    """Enregistre une sortie Replicate (FileOutput ou URL) dans `chemin`.

    Ne pas utiliser `sortie.read()` : le SDK envoie l'en-tête Authorization au
    stockage des fichiers générés, qui répond 400. On télécharge l'URL sans clé.
    """
    url = getattr(sortie, "url", None) or str(sortie)
    r = httpx.get(url, follow_redirects=True, timeout=120)
    r.raise_for_status()
    with open(chemin, "wb") as f:
        f.write(r.content)
    return chemin
