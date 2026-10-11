from tqdm import tqdm
from deep_translator import MyMemoryTranslator
from deep_translator.exceptions import TooManyRequests
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time
import os
import requests

MYMEMORY_MAX_CHARS = 490

# MyMemory requires region-tagged codes (e.g. "es-ES"), unlike Google's
# bare ISO codes (e.g. "es") used throughout this app.
MYMEMORY_LANG_CODES = {
    "auto": "auto",
    "ar": "ar-SA", "zh": "zh-CN", "zh-TW": "zh-TW", "cs": "cs-CZ",
    "da": "da-DK", "nl": "nl-NL", "en": "en-US", "fi": "fi-FI",
    "fr": "fr-FR", "de": "de-DE", "el": "el-GR", "he": "he-IL",
    "iw": "he-IL", "hu": "hu-HU", "it": "it-IT", "ja": "ja-JP",
    "ko": "ko-KR", "fa": "fa-IR", "pl": "pl-PL", "pt": "pt-PT",
    "ru": "ru-RU", "es": "es-ES", "tr": "tr-TR", "uk": "uk-UA",
    "ur": "ur-PK", "vi": "vi-VN", "hi": "hi-IN", "id": "id-ID",
    "bn": "bn-IN", "te": "te-IN", "mr": "mr-IN", "ta": "ta-IN",
    "ca": "ca-ES", "ne": "ne-NP", "th": "th-TH", "sv": "sv-SE",
    "am": "am-ET", "cy": "cy-GB", "et": "et-EE", "hr": "hr-HR",
    "is": "is-IS", "ka": "ka-GE", "km": "km-KH", "sk": "sk-SK",
    "sq": "sq-AL", "sr": "sr-Cyrl-RS", "az": "az-AZ", "bg": "bg-BG",
    "gl": "gl-ES", "gu": "gu-IN", "kk": "kk-KZ", "kn": "kn-IN",
    "lt": "lt-LT", "lv": "lv-LV", "mk": "mk-MK", "ml": "ml-IN",
    "ms": "ms-MY", "ro": "ro-RO", "si": "si-LK", "su": "su-ID",
    "sw": "sw-KE", "af": "af-ZA", "bs": "bs-BA", "my": "my-MM",
    "no": "nb-NO", "as": "as-IN", "eu": "eu-ES",
}


def _to_mymemory_code(lang_code):
    return MYMEMORY_LANG_CODES.get(lang_code, lang_code)


def _mymemory_translate(source, target, text):
    """
    Translates text with MyMemory, splitting on word boundaries to
    respect its 500-char per-request limit.
    """
    text = text.strip()
    if not text:
        return text

    translator = MyMemoryTranslator(
        source=_to_mymemory_code(source),
        target=_to_mymemory_code(target),
        email="shivrajsingh1059@gmail.com",
    )

    if len(text) <= MYMEMORY_MAX_CHARS:
        return translator.translate(text)

    words = text.split(" ")
    pieces = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= MYMEMORY_MAX_CHARS:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = word
    if current:
        pieces.append(current)

    return " ".join(translator.translate(piece) for piece in pieces)


def _mymemory_translate_with_retry(source, target, text, max_retries=2, base_delay=3):
    """
    Calls _mymemory_translate(), retrying briefly on MyMemory's own
    rate limit (a burst/short-term throttle, separate from its daily
    word quota) before giving up and keeping the original text.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return _mymemory_translate(source, target, text)
        except TooManyRequests as error:
            last_error = error
            delay = base_delay * (attempt + 1)
            logger.warning(
                f"MyMemory rate limit hit, retrying in {delay}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
        except Exception as error:
            last_error = error
            break

    logger.warning(
        "MyMemory translator also failed/rate-limited, keeping "
        f"the original text for this line. Related: {last_error}"
    )
    return text

from concurrent.futures import ThreadPoolExecutor

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "deepl",
    "mymemory",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    "disable_translation",
]
DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "deepl",
    "mymemory",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "disable_translation",
]


def translate_iterative(segments, target, source=None):
    """
    Translate text segments concurrently to the specified language.
    """
    segments_ = copy.deepcopy(segments)

    if not source:
        logger.debug("No source language")
        source = "auto"

    def _translate_one(seg):
        text = seg["text"].strip()
        if not text:
            return ""
        return _mymemory_translate_with_retry(source, target, text)

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(tqdm(executor.map(_translate_one, segments_), total=len(segments_), desc="Translating (Parallel)"))

    for i, res in enumerate(results):
        segments_[i]["text"] = res

    return segments_


def verify_translate(
    segments,
    segments_copy,
    translated_lines,
    target,
    source
):
    """
    Verify integrity and translate segments if lengths match, otherwise
    switch to iterative translation.
    """
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            logger.debug(
                f"{segments_copy[line]['text']} >> "
                f"{translated_lines[line].strip()}"
            )
            segments_copy[line]["text"] = translated_lines[
                line].replace("\t", "").replace("\n", "").strip()
        return segments_copy
    else:
        logger.error(
            "The translation failed, switching to MyMemory iterative. "
            f"{len(segments), len(translated_lines)}"
        )
        return translate_iterative(segments, target, source)


def _quality_check_translation(original_text, translated_text, start=None):
    """
    Lightweight, non-blocking safeguards for LLM translations: flags
    (via warning logs, without altering the result) lines that are
    disproportionately long for subtitle/dubbing timing, or that drop
    numbers present in the source - both common LLM translation failure
    modes.
    """
    if not original_text or not translated_text:
        return

    where = f" (segment {start})" if start is not None else ""

    orig_len = len(original_text)
    trans_len = len(translated_text)
    if orig_len > 0 and trans_len > orig_len * 1.8 and trans_len > 60:
        logger.warning(
            f"Translation quality check{where}: translated line is "
            f"much longer than the original ({trans_len} vs {orig_len} "
            "chars) - may not fit the original timing."
        )

    original_numbers = set(re.findall(r"\d+", original_text))
    translated_numbers = set(re.findall(r"\d+", translated_text))
    missing_numbers = original_numbers - translated_numbers
    if missing_numbers:
        logger.warning(
            f"Translation quality check{where}: number(s) "
            f"{sorted(missing_numbers)} from the source line are missing "
            "in the translation - please verify."
        )


def call_gpt_translate(
    client,
    model,
    system_prompt,
    user_prompt,
    original_text=None,
    batch_lines=None,
):

    # https://platform.openai.com/docs/guides/text-generation/json-mode
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content
    logger.debug(f"Result: {str(result)}")

    try:
        translation = json.loads(result)
    except Exception as error:
        match_result = re.search(r'\{.*?\}', result)
        if match_result:
            logger.error(str(error))
            json_str = match_result.group(0)
            translation = json.loads(json_str)
        else:
            raise error

    # Get valid data
    if batch_lines:
        for conversation in translation.values():
            if isinstance(conversation, dict):
                conversation = list(conversation.values())[0]
            if (
                list(
                    original_text["conversation"][0].values()
                )[0].strip() ==
                list(conversation[0].values())[0].strip()
            ):
                continue
            if len(conversation) == batch_lines:
                break

        fix_conversation_length = []
        for line in conversation:
            for speaker_code, text_tr in line.items():
                fix_conversation_length.append({speaker_code: text_tr})

        logger.debug(f"Data batch: {str(fix_conversation_length)}")
        logger.debug(
            f"Lines Received: {len(fix_conversation_length)},"
            f" expected: {batch_lines}"
        )

        return fix_conversation_length

    else:
        if isinstance(translation, dict):
            translation = list(translation.values())[0]
        if isinstance(translation, list):
            translation = translation[0]
        if isinstance(translation, set):
            translation = list(translation)[0]
        if not isinstance(translation, str):
            raise ValueError(f"No valid response received: {str(translation)}")

        return translation


def gpt_sequential(segments, model, target, source=None):
    from openai import OpenAI

    translated_segments = copy.deepcopy(segments)

    client = OpenAI()
    progress_bar = tqdm(total=len(segments), desc="Translating")

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    system_prompt = (
        "You are a professional native-English translator and localization "
        "expert preparing dubbing/subtitle scripts for a video. Translate "
        "by meaning and context, never word-for-word. Follow these rules "
        "strictly:\n"
        "1. Produce natural, fluent, spoken English a native speaker would "
        "actually say out loud - not a literal or machine-sounding "
        "translation.\n"
        "2. Preserve the speaker's original meaning, tone, emotion, and "
        "intent (formal stays formal, casual stays conversational, humor "
        "stays humor).\n"
        "3. Correctly interpret idioms and expressions instead of "
        "translating them literally (e.g. \"esta lloviendo a cantaros\" -> "
        "\"it's pouring\", not \"it's raining jugs\").\n"
        "4. Never invent information that wasn't said, never drop "
        "meaningful information, never summarize, and never skip words or "
        "sentences - translate the complete line.\n"
        "5. If part of the line is already in English, leave that part "
        "unchanged instead of translating it.\n"
        "6. If a line is unclear or ambiguous, translate it as literally "
        "and faithfully as possible rather than guessing or interpreting "
        "it creatively.\n"
        "7. If the speaker repeats a word or phrase, preserve that "
        "repetition in the translation when it reads naturally.\n"
        "8. Never translate people's names. Keep brand/product names, "
        "places, and other proper nouns unchanged. Keep numbers and dates "
        "exactly as stated.\n"
        "9. Use natural contractions where a native speaker would (I'm, "
        "don't, we're, etc.).\n"
        "10. Keep the translation reasonably close in length to the "
        "original line so it still fits the same subtitle/audio timing; "
        "avoid unnecessarily long sentences.\n"
        "11. You may be given the previous and next lines purely as "
        "context to understand meaning and keep terminology consistent - "
        "translate and return ONLY the current line, not the context "
        "lines.\n"
        'Respond with a JSON object: {"translated_text": "..."}'
    )

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        start = line["start"]

        context_parts = []
        if i > 0:
            context_parts.append(
                f"- Previous line (original): {segments[i - 1]['text'].strip()}"
            )
            context_parts.append(
                "- Previous line (already translated, for terminology "
                f"consistency): {translated_segments[i - 1]['text'].strip()}"
            )
        if i < len(segments) - 1:
            context_parts.append(
                f"- Next line (original): {segments[i + 1]['text'].strip()}"
            )
        context_block = (
            "\nContext only, do not translate these lines:\n"
            + "\n".join(context_parts) + "\n"
            if context_parts else ""
        )

        user_prompt = (
            f"Translate the following{f' {lang_sc}' if lang_sc else ''} "
            f"line into natural, spoken {lang_tg}, matching the speaker's "
            f"tone and intent.{context_block}\n"
            f"Line to translate: {text}"
        )

        time.sleep(0.5)

        try:
            translated_text = call_gpt_translate(
                client,
                model,
                system_prompt,
                user_prompt,
            )
            _quality_check_translation(text, translated_text.strip(), start)

        except Exception as error:
            logger.error(
                f"{str(error)} >> The text of segment {start} "
                "is being corrected with MyMemory"
            )
            translated_text = _mymemory_translate_with_retry(
                fixed_source, fixed_target, text.strip()
            )
            time.sleep(0.4)

        translated_segments[i]["text"] = translated_text.strip()
        progress_bar.update(1)

    progress_bar.close()

    return translated_segments


def gpt_batch(segments, model, target, token_batch_limit=900, source=None):
    from openai import OpenAI
    import tiktoken

    token_batch_limit = max(100, (token_batch_limit - 40) // 2)
    progress_bar = tqdm(total=len(segments), desc="Translating")
    segments_copy = copy.deepcopy(segments)
    encoding = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[target]).strip()
    lang_sc = ""
    if source:
        lang_sc = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES[source]).strip()

    fixed_target = fix_code_language(target)
    fixed_source = fix_code_language(source) if source else "auto"

    name_speaker = "ABCDEFGHIJKL"

    translated_lines = []
    text_data_dict = []
    num_tokens = 0
    count_sk = {char: 0 for char in "ABCDEFGHIJKL"}
    previous_context = None

    for i, line in enumerate(segments_copy):
        text = line["text"]
        speaker = line["speaker"]
        last_start = line["start"]
        # text_data_dict.append({str(int(speaker[-1])+1): text})
        index_sk = int(speaker[-2:])
        character_sk = name_speaker[index_sk]
        count_sk[character_sk] += 1
        code_sk = character_sk+str(count_sk[character_sk])
        text_data_dict.append({code_sk: text})
        num_tokens += len(encoding.encode(text)) + 7
        if num_tokens >= token_batch_limit or i == len(segments_copy)-1:
            try:
                batch_lines = len(text_data_dict)
                batch_conversation = {"conversation": copy.deepcopy(text_data_dict)}
                # Reset vars
                num_tokens = 0
                text_data_dict = []
                count_sk = {char: 0 for char in "ABCDEFGHIJKL"}
                # Process translation
                # https://arxiv.org/pdf/2309.03409.pdf
                system_prompt = (
                    "You are a professional native-English translator and "
                    "localization expert preparing dubbing/subtitle scripts "
                    "for a video. You receive a JSON object with a "
                    f"'conversation' list of {batch_lines} sequential "
                    f"dialogue lines{f' spoken in {lang_sc}' if lang_sc else ''}"
                    ", each item keyed by a line id (e.g. \"A1\"). Translate "
                    "the ENTIRE conversation as one continuous piece, using "
                    "every line as context for the others, into natural, "
                    f"spoken {lang_tg} - the way a professional human "
                    "translator would write it for a native speaker, never "
                    "a literal or word-for-word translation. Rules:\n"
                    "- Preserve the meaning, tone, emotion, and intent of "
                    "each line (formal stays formal, casual stays "
                    "conversational, jokes stay jokes).\n"
                    "- Correctly interpret idioms/expressions instead of "
                    "translating them literally.\n"
                    "- Never invent information that wasn't said, never "
                    "drop meaningful information, never summarize, and "
                    "never skip words or sentences - translate each line "
                    "completely.\n"
                    "- If part of a line is already in English, leave "
                    "that part unchanged instead of translating it.\n"
                    "- If a line is unclear or ambiguous, translate it as "
                    "literally and faithfully as possible rather than "
                    "guessing or interpreting it creatively.\n"
                    "- If the speaker repeats a word or phrase, preserve "
                    "that repetition in the translation when it reads "
                    "naturally.\n"
                    "- Never translate people's names. Keep brand/product "
                    "names, places, and other proper nouns unchanged. Keep "
                    "numbers and dates exactly as stated.\n"
                    "- Keep terminology consistent across all lines.\n"
                    "- Use natural contractions where a native speaker "
                    "would (I'm, don't, we're, etc.).\n"
                    "- Keep each translated line reasonably close in "
                    "length to its original so it still fits the same "
                    "subtitle/audio timing; avoid unnecessarily long "
                    "sentences.\n"
                    "- Preserve the exact line order and the exact same "
                    "line ids; never merge, split, add, or remove lines.\n"
                    'Respond with a JSON object: {"translated_conversation": '
                    '[{"<line id>": "..."}, ...]} containing exactly '
                    f"{batch_lines} items in the same order."
                )
                context_block = (
                    "\nFor context only (already translated, keep "
                    "terminology/tone consistent with it, do not "
                    f"re-translate it): {previous_context}\n"
                    if previous_context else ""
                )
                user_prompt = (
                    "Translate this conversation"
                    f"{f' from {lang_sc}' if lang_sc else ''} to {lang_tg}, "
                    "keeping the meaning, tone, and natural flow of a real "
                    f"conversation.{context_block}\n{batch_conversation}"
                )
                logger.debug(f"Prompt: {str(user_prompt)}")

                conversation = call_gpt_translate(
                    client,
                    model,
                    system_prompt,
                    user_prompt,
                    original_text=batch_conversation,
                    batch_lines=batch_lines,
                )

                if len(conversation) < batch_lines:
                    raise ValueError(
                        "Incomplete result received. Batch lines: "
                        f"{len(conversation)}, expected: {batch_lines}"
                    )

                batch_sources = batch_conversation["conversation"]
                for i, translated_text in enumerate(conversation):
                    if i+1 > batch_lines:
                        break
                    translated_value = list(translated_text.values())[0]
                    _quality_check_translation(
                        list(batch_sources[i].values())[0], translated_value
                    )
                    translated_lines.append(translated_value)

                previous_context = translated_lines[-1]
                progress_bar.update(batch_lines)

            except Exception as error:
                logger.error(str(error))

                first_start = segments_copy[max(0, i-(batch_lines-1))]["start"]
                logger.warning(
                    f"The batch from {first_start} to {last_start} "
                    "failed, is being corrected with MyMemory"
                )

                for txt_source in batch_conversation["conversation"]:
                    translated_txt = _mymemory_translate_with_retry(
                        fixed_source,
                        fixed_target,
                        list(txt_source.values())[0].strip(),
                    )
                    translated_lines.append(translated_txt.strip())
                    progress_bar.update(1)
                    time.sleep(0.4)

    progress_bar.close()

    return verify_translate(
        segments, segments_copy, translated_lines, fixed_target, fixed_source
    )


DEEPL_ENV_VAR = "DEEPL_API_KEY"


def _to_deepl_code(code, is_target):
    """
    DeepL expects uppercase codes, and for a handful of languages the
    TARGET code needs a region variant (source doesn't).
    """
    code = code.upper()
    if not is_target:
        return code
    if code == "EN":
        return "EN-US"
    if code == "PT":
        return "PT-BR"
    if code == "ZH":
        return "ZH-HANS"
    return code


def translate_google_batch(segments, target, source=None, max_chars=4000):
    """
    Lightning-fast batch translation using Google's translate_a API.
    Translates segments joined by delimiter in bulk HTTP requests.
    Translates hundreds of lines in under 1 second without rate limits.
    """
    segments_copy = copy.deepcopy(segments)
    target_lang = fix_code_language(target)
    source_lang = fix_code_language(source) if source else "auto"

    DELIMITER = "\n||||\n"
    translated_lines = []

    chunks = []
    current_chunk = []
    current_length = 0

    for seg in segments_copy:
        text = seg["text"].strip() or " "
        line_len = len(text) + len(DELIMITER)
        if current_length + line_len > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [text]
            current_length = line_len
        else:
            current_chunk.append(text)
            current_length += line_len
    if current_chunk:
        chunks.append(current_chunk)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    progress_bar = tqdm(total=len(segments_copy), desc="Translating (Google Fast Batch)")
    for chunk in chunks:
        joined_text = DELIMITER.join(chunk)
        chunk_translated = False
        try:
            resp = session.get(
                "https://translate.googleapis.com/translate_a/single",
                params={
                    "client": "gtx",
                    "sl": source_lang,
                    "tl": target_lang,
                    "dt": "t",
                    "q": joined_text,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                full_translated = "".join(part[0] for part in data[0] if part and part[0])
                split_translated = full_translated.split("||||")
                if len(split_translated) == len(chunk):
                    translated_lines.extend(t.strip() for t in split_translated)
                    chunk_translated = True
        except Exception as err:
            logger.debug(f"Google delimiter batch failed: {err}")

        if not chunk_translated:
            # Parallel line-by-line fallback for this chunk
            def _trans_line(t):
                if not t.strip():
                    return ""
                try:
                    r = session.get(
                        "https://translate.googleapis.com/translate_a/single",
                        params={"client": "gtx", "sl": source_lang, "tl": target_lang, "dt": "t", "q": t},
                        timeout=10,
                    )
                    return "".join(p[0] for p in r.json()[0] if p and p[0]).strip()
                except Exception:
                    return t
            with ThreadPoolExecutor(max_workers=6) as ex:
                translated_lines.extend(list(ex.map(_trans_line, chunk)))

        progress_bar.update(len(chunk))

    progress_bar.close()

    return verify_translate(
        segments, segments_copy, translated_lines,
        target_lang, source_lang
    )


def translate_deepl_batch(segments, target, source=None, batch_size=50):
    """
    Translate segments using the DeepL API (real per-request batching).
    Requires a DEEPL_API_KEY environment variable.
    Falls back to translate_google_batch if no key or DeepL fails.
    """
    api_key = os.environ.get(DEEPL_ENV_VAR)
    if not api_key:
        logger.warning(
            f"No {DEEPL_ENV_VAR} set, falling back to Google Fast Batch"
        )
        return translate_google_batch(segments, target, source)

    segments_copy = copy.deepcopy(segments)
    base_url = (
        "https://api-free.deepl.com/v2/translate"
        if api_key.endswith(":fx")
        else "https://api.deepl.com/v2/translate"
    )
    target_lang = _to_deepl_code(fix_code_language(target), is_target=True)
    source_lang = (
        _to_deepl_code(fix_code_language(source), is_target=False)
        if source
        else None
    )

    text_lines = [line["text"].strip() or " " for line in segments_copy]
    translated_lines = []
    progress_bar = tqdm(total=len(text_lines), desc="Translating (DeepL Batch)")
    session = requests.Session()

    for i in range(0, len(text_lines), batch_size):
        chunk = text_lines[i:i + batch_size]
        data = [("text", t) for t in chunk]
        data.append(("target_lang", target_lang))
        if source_lang:
            data.append(("source_lang", source_lang))

        last_error = None
        for attempt in range(3):
            try:
                response = session.post(
                    base_url,
                    headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
                    data=data,
                    timeout=30,
                )
                if response.status_code == 429 or response.status_code == 456:
                    raise TooManyRequests()
                response.raise_for_status()
                result = response.json()
                translated_lines.extend(
                    t["text"] for t in result["translations"]
                )
                last_error = None
                break
            except TooManyRequests as error:
                last_error = error
                delay = 2 * (attempt + 1)
                logger.warning(
                    f"DeepL rate limit hit, retrying in {delay}s "
                    f"(attempt {attempt + 1}/3)"
                )
                time.sleep(delay)
            except Exception as error:
                last_error = error
                break

        if last_error is not None:
            logger.warning(
                "DeepL translation failed for this chunk, using Google Fast Batch fallback. "
                f"Related: {last_error}"
            )
            fallback_segs = [{"text": t} for t in chunk]
            fallback_res = translate_google_batch(fallback_segs, target, source)
            translated_lines.extend([f["text"] for f in fallback_res])

        progress_bar.update(len(chunk))

    progress_bar.close()

    return verify_translate(
        segments, segments_copy, translated_lines,
        fix_code_language(target), fix_code_language(source)
    )


def translate_text(
    segments,
    target,
    translation_process="google_translator_batch",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
):
    """Translates text segments using a specified process."""
    match translation_process:
        case "google_translator_batch":
            return translate_google_batch(segments, target, source)
        case "deepl":
            return translate_deepl_batch(segments, target, source)
        case "mymemory":
            return translate_iterative(
                segments,
                fix_code_language(target),
                fix_code_language(source)
            )
        case model if model in ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]:
            return gpt_sequential(segments, model, target, source)
        case model if model in ["gpt-3.5-turbo-0125_batch", "gpt-4-turbo-preview_batch",]:
            return gpt_batch(
                segments,
                translation_process.replace("_batch", ""),
                target,
                token_batch_limit,
                source
            )
        case "disable_translation":
            return segments
        case _:
            raise ValueError("No valid translation process")
