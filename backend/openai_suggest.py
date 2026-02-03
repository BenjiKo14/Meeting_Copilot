from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Suggestions:
    best: str
    alternative: str  # Single longer/more complete alternative
    clarifying_question: Optional[str]
    technical_terms: list[dict[str, str]]  # List of {term, explanation}


def _system_prompt(lang: str, profile: str = "") -> str:
    profile_section = ""
    if profile:
        if lang == "en":
            profile_section = (
                f"\n\n=== CANDIDATE PROFILE (ME) ===\n{profile}\n"
                "=== END PROFILE ===\n\n"
            )
        else:
            profile_section = (
                f"\n\n=== PROFIL DU CANDIDAT (ME) ===\n{profile}\n"
                "=== FIN PROFIL ===\n\n"
            )
    
    if lang == "en":
        base = (
            "You are a real-time interview assistant. You help ME reply to OTHER (recruiter/employer).\n\n"
            f"{profile_section}"
            "🚨 CRITICAL RULES 🚨\n\n"
            "1. ALWAYS ANSWER TECHNICAL QUESTIONS\n"
            "   - ME is intelligent and knowledgeable. NEVER say 'this is not my domain' or 'I'm not specialized in this'.\n"
            "   - If OTHER asks a technical question (algorithms, systems, theory), ANSWER IT correctly and confidently.\n"
            "   - ME knows computer science, AI/ML, systems, and can explain complex concepts.\n"
            "   - Connect the answer to the profile when relevant, but ALWAYS provide a real answer.\n\n"
            "2. NEVER LIE ABOUT PERSONAL EXPERIENCE\n"
            "   - Don't invent projects, achievements, or metrics that aren't in the profile.\n"
            "   - Don't say 'I reduced delays by X%' or 'I managed a team of X' unless it's IN THE PROFILE.\n"
            "   - Use ONLY real data from the profile provided above.\n\n"
            "3. DISTINCTION\n"
            "   - General knowledge (how does virtual memory work?) = ANSWER FULLY, ME knows this.\n"
            "   - Personal experience (what projects have you done?) = ONLY use facts from profile.\n\n"
            "Tone: confident, competent, professional. Help ME get the job.\n"
            "Do not mention you are an AI."
        )
    else:
        base = (
            "Tu es un assistant d'entretien en temps réel. Tu aides ME à répondre à OTHER (recruteur/employeur).\n\n"
            f"{profile_section}"
            "🚨 RÈGLES CRITIQUES 🚨\n\n"
            "1. TOUJOURS RÉPONDRE AUX QUESTIONS TECHNIQUES\n"
            "   - ME est intelligent et cultivé. JAMAIS dire 'ce n'est pas mon domaine' ou 'je ne suis pas spécialisé'.\n"
            "   - Si OTHER pose une question technique (algorithmes, systèmes, théorie), RÉPONDS correctement et avec confiance.\n"
            "   - ME connaît l'informatique, l'IA/ML, les systèmes, et peut expliquer des concepts complexes.\n"
            "   - Connecte la réponse au profil quand c'est pertinent, mais FOURNIS TOUJOURS une vraie réponse.\n\n"
            "2. JAMAIS MENTIR SUR L'EXPÉRIENCE PERSONNELLE\n"
            "   - N'invente pas de projets, réalisations ou métriques qui ne sont pas dans le profil.\n"
            "   - Ne dis pas 'j'ai réduit les délais de X%' ou 'j'ai géré une équipe de X' sauf si c'est DANS LE PROFIL.\n"
            "   - Utilise UNIQUEMENT les vraies données du profil fourni ci-dessus.\n\n"
            "3. DISTINCTION\n"
            "   - Connaissances générales (comment fonctionne la mémoire virtuelle?) = RÉPONDS COMPLÈTEMENT, ME sait ça.\n"
            "   - Expérience personnelle (quels projets as-tu faits?) = UNIQUEMENT les faits du profil.\n\n"
            "Ton: confiant, compétent, professionnel. Aide ME à décrocher le poste.\n"
            "Ne mentionne pas que tu es une IA."
        )
    return base


def _user_prompt(lang: str, transcript: str) -> str:
    if lang == "en":
        return (
            "Language: en\n"
            "Conversation transcript (oldest to newest):\n"
            f"{transcript}\n\n"
            "IMPORTANT: Focus on what OTHER (the interviewer/recruiter) just said or asked. "
            "Propose responses for ME (the candidate) to answer OTHER's questions or comments. "
            "Use what ME said before for context, but respond to OTHER.\n\n"
            "Tasks:\n"
            "1. Propose 1 BEST reply (concise, 2-3 sentences)\n"
            "2. Propose 1 ALTERNATIVE reply that is MORE COMPLETE and LONGER (4-6 sentences, more detailed)\n"
            "3. If the question is ambiguous, propose a clarifying question\n"
            "4. Extract ONLY technical terms that are ACTUALLY WRITTEN in the transcript above. "
            "Do NOT invent terms. Only include words/acronyms that appear verbatim in the conversation "
            "(like MLOps, Kubernetes, RAG, CI/CD, Docker, etc.). If no technical terms appear, return empty array.\n\n"
            "Return STRICT JSON with keys:\n"
            "- best (string): concise best answer\n"
            "- alternative (string): longer, more detailed answer\n"
            "- clarifying_question (string|null)\n"
            "- technical_terms (array of objects with 'term' and 'explanation' - ONLY terms actually in the transcript)"
        )
    return (
        "Langue: fr\n"
        "Transcription de la conversation (du plus ancien au plus récent):\n"
        f"{transcript}\n\n"
        "IMPORTANT: Concentre-toi sur ce que OTHER (le recruteur/intervieweur) vient de dire ou demander. "
        "Propose des réponses pour ME (le candidat) pour répondre aux questions ou commentaires de OTHER. "
        "Utilise ce que ME a dit avant pour le contexte, mais réponds à OTHER.\n\n"
        "Tâches:\n"
        "1. Propose 1 MEILLEURE réponse (concise, 2-3 phrases)\n"
        "2. Propose 1 réponse ALTERNATIVE plus COMPLÈTE et LONGUE (4-6 phrases, plus détaillée)\n"
        "3. Si la question est ambiguë, propose une question de clarification\n"
        "4. Extrais UNIQUEMENT les termes techniques qui sont RÉELLEMENT ÉCRITS dans la transcription ci-dessus. "
        "N'invente PAS de termes. Inclus seulement les mots/acronymes qui apparaissent textuellement dans la conversation "
        "(comme MLOps, Kubernetes, RAG, CI/CD, Docker, etc.). Si aucun terme technique n'apparaît, retourne un tableau vide.\n\n"
        "Retourne du JSON STRICT avec les clés:\n"
        "- best (string): meilleure réponse concise\n"
        "- alternative (string): réponse plus longue et détaillée\n"
        "- clarifying_question (string|null)\n"
        "- technical_terms (array d'objets avec clés 'term' et 'explanation' - UNIQUEMENT les termes présents dans la transcription)"
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # crude fallback: grab first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _normalize(obj: dict[str, Any]) -> Suggestions:
    best = str(obj.get("best") or "").strip()
    
    # Single alternative (longer/more complete)
    alternative = str(obj.get("alternative") or "").strip()
    
    # Clarifying question
    cq = obj.get("clarifying_question")
    cq_s = None
    if cq is not None:
        cq_s = str(cq).strip() or None
    
    # Technical terms explanations
    terms_raw = obj.get("technical_terms") or []
    if not isinstance(terms_raw, list):
        terms_raw = []
    technical_terms = []
    for t in terms_raw:
        if isinstance(t, dict):
            term = str(t.get("term") or "").strip()
            explanation = str(t.get("explanation") or "").strip()
            if term and explanation:
                technical_terms.append({"term": term, "explanation": explanation})
    
    return Suggestions(
        best=best, 
        alternative=alternative, 
        clarifying_question=cq_s,
        technical_terms=technical_terms
    )


def generate_chat_response(*, lang: str, transcript: str, profile: str = "", question: str) -> str:
    """
    Answer a free-form question about the conversation.
    """
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    if lang == "en":
        system = (
            "You are a helpful interview assistant. You have access to the conversation transcript and the candidate's profile.\n"
            "Answer the user's question based on this context. Be concise and helpful.\n"
            "If the question is about strategy, give practical advice for the interview.\n"
            f"\n=== CANDIDATE PROFILE ===\n{profile}\n=== END PROFILE ===\n" if profile else ""
        )
        user_msg = f"Conversation transcript:\n{transcript}\n\nQuestion: {question}"
    else:
        system = (
            "Tu es un assistant d'entretien. Tu as accès à la transcription de la conversation et au profil du candidat.\n"
            "Réponds à la question de l'utilisateur en te basant sur ce contexte. Sois concis et utile.\n"
            "Si la question concerne la stratégie, donne des conseils pratiques pour l'entretien.\n"
            f"\n=== PROFIL DU CANDIDAT ===\n{profile}\n=== FIN PROFIL ===\n" if profile else ""
        )
        user_msg = f"Transcription de la conversation:\n{transcript}\n\nQuestion: {question}"

    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )

    return (resp.choices[0].message.content or "").strip()


def generate_suggestions(*, lang: str, transcript: str, profile: str = "") -> Suggestions:
    """
    Synchronous helper. Called from FastAPI threadpool (to avoid blocking event loop).
    Uses OpenAI Chat Completions with JSON output.
    """
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    # Local import so the module still loads even if dependency missing.
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _system_prompt(lang, profile)},
            {"role": "user", "content": _user_prompt(lang, transcript)},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()
    data = _extract_json(content)
    return _normalize(data)

