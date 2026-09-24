from tqdm import tqdm
from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.exceptions import TooManyRequests
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time

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


_google_cooldown_until = 0.0
_GOOGLE_COOLDOWN_SECONDS = 60


def translate_with_retry(
    translator, text, source, target, max_retries=3, base_delay=2
):
    """
    Calls translator.translate(), retrying briefly when Google's free
    endpoint rate-limits the request. If Google keeps refusing (e.g. the
    IP got blocked rather than just rate-limited), falls back to the
    MyMemory translator instead of failing the whole job.

    Once Google refuses a request, a short-lived cooldown is set so
    later calls (other lines/chunks in the same job) skip straight to
    MyMemory instead of each burning time on retries against an
    endpoint that's already known to be blocked.
    """
    global _google_cooldown_until

    if time.time() < _google_cooldown_until:
        try:
            return _mymemory_translate(source, target, text)
        except Exception as error:
            logger.warning(
                "Google Translate still in cooldown and MyMemory also "
                f"failed, keeping the original text for this line. "
                f"Related: {error}"
            )
            return text

    last_error = None
    for attempt in range(max_retries):
        try:
            return translator.translate(text)
        except TooManyRequests as error:
            last_error = error
            _google_cooldown_until = time.time() + _GOOGLE_COOLDOWN_SECONDS
            logger.warning(
                "Google Translate rate limit hit, switching to MyMemory "
                f"for the next {_GOOGLE_COOLDOWN_SECONDS}s"
            )
            break

    logger.warning(
        "Google Translate keeps rate-limiting this request, falling "
        f"back to MyMemory translator. Related: {last_error}"
    )
    try:
        return _mymemory_translate(source, target, text)
    except Exception as error:
        logger.warning(
            "MyMemory translator also failed/rate-limited, keeping "
            f"the original text for this line. Related: {error}"
        )
        return text

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    "disable_translation",
]
DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "disable_translation",
]


def translate_iterative(segments, target, source=None):
    """
    Translate text segments individually to the specified language.

    Parameters:
    - segments (list): A list of dictionaries with 'text' as a key for
        segment text.
    - target (str): Target language code.
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Translates each segment using Google Translate.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated_segments = translate_iterative(segments, 'es')
    """

    segments_ = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    translator = GoogleTranslator(source=source, target=target)

    for line in tqdm(range(len(segments_))):
        text = segments_[line]["text"]
        translated_line = translate_with_retry(
            translator, text.strip(), source, target
        )
        segments_[line]["text"] = translated_line
        time.sleep(0.25)

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
            "The translation failed, switching to google_translate iterative. "
            f"{len(segments), len(translated_lines)}"
        )
        return translate_iterative(segments, target, source)


def translate_batch(segments, target, chunk_size=2000, source=None):
    """
    Translate a batch of text segments into the specified language in chunks,
        respecting the character limit.

    Parameters:
    - segments (list): List of dictionaries with 'text' as a key for segment
        text.
    - target (str): Target language code.
    - chunk_size (int, optional): Maximum character limit for each translation
        chunk (default is 2000; max 5000).
    - source (str, optional): Source language code. Defaults to None.

    Returns:
    - list: Translated text segments in the target language.

    Notes:
    - Splits input segments into chunks respecting the character limit for
        translation.
    - Translates the chunks using Google Translate.
    - If chunked translation fails, switches to iterative translation using
        `translate_iterative()`.

    Example:
    segments = [{'text': 'first segment.'}, {'text': 'second segment.'}]
    translated = translate_batch(segments, 'es', chunk_size=4000, source='en')
    """

    segments_copy = copy.deepcopy(segments)

    if (
        not source
    ):
        logger.debug("No source language")
        source = "auto"

    # Get text
    text_lines = []
    for line in range(len(segments_copy)):
        text = segments_copy[line]["text"].strip()
        text_lines.append(text)

    # chunk limit
    text_merge = []
    actual_chunk = ""
    global_text_list = []
    actual_text_list = []
    for one_line in text_lines:
        one_line = " " if not one_line else one_line
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk:
                actual_chunk += " ||||| "
            actual_chunk += one_line
            actual_text_list.append(one_line)
        else:
            text_merge.append(actual_chunk)
            actual_chunk = one_line
            global_text_list.append(actual_text_list)
            actual_text_list = [one_line]
    if actual_chunk:
        text_merge.append(actual_chunk)
        global_text_list.append(actual_text_list)

    # translate chunks
    progress_bar = tqdm(total=len(segments), desc="Translating")
    translator = GoogleTranslator(source=source, target=target)
    split_list = []
    try:
        for text, text_iterable in zip(text_merge, global_text_list):
            translated_line = translate_with_retry(
                translator, text.strip(), source, target
            )
            split_text = translated_line.split("|||||")
            if len(split_text) == len(text_iterable):
                progress_bar.update(len(split_text))
            else:
                logger.debug(
                    "Chunk fixing iteratively. Len chunk: "
                    f"{len(split_text)}, expected: {len(text_iterable)}"
                )
                split_text = []
                for txt_iter in text_iterable:
                    translated_txt = translate_with_retry(
                        translator, txt_iter.strip(), source, target
                    )
                    split_text.append(translated_txt)
                    progress_bar.update(1)
                    time.sleep(0.25)
            split_list.append(split_text)
            time.sleep(0.25)
        progress_bar.close()
    except Exception as error:
        progress_bar.close()
        logger.error(str(error))
        logger.warning(
            "The translation in chunks failed, switching to iterative."
            " Related: too many request"
        )  # use proxy or less chunk size
        return translate_iterative(segments, target, source)

    # un chunk
    translated_lines = list(chain.from_iterable(split_list))

    return verify_translate(
        segments, segments_copy, translated_lines, target, source
    )


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
                "is being corrected with Google Translate"
            )
            translator = GoogleTranslator(
                source=fixed_source, target=fixed_target
            )
            translated_text = translate_with_retry(
                translator, text.strip(), fixed_source, fixed_target
            )
            time.sleep(0.25)

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
                    "failed, is being corrected with Google Translate"
                )

                translator = GoogleTranslator(
                    source=fixed_source,
                    target=fixed_target
                )

                for txt_source in batch_conversation["conversation"]:
                    translated_txt = translate_with_retry(
                        translator,
                        list(txt_source.values())[0].strip(),
                        fixed_source,
                        fixed_target,
                    )
                    translated_lines.append(translated_txt.strip())
                    progress_bar.update(1)
                    time.sleep(0.25)

    progress_bar.close()

    return verify_translate(
        segments, segments_copy, translated_lines, fixed_target, fixed_source
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
            return translate_batch(
                segments,
                fix_code_language(target),
                chunk_size,
                fix_code_language(source)
            )
        case "google_translator":
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
