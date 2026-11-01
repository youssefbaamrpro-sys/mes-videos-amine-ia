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

GROQ_MODEL = "openai/gpt-oss-120b"
