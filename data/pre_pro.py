# encoding: utf8

import pandas as pd
import re


def heb(s, t): return ('HEB', t)


def eng(s, t): return ('ENG', t)


def num(s, t): return ('NUM', t)


def url(s, t): return ('URL', t)


def punct(s, t): return ('PUNCT', t)


def junk(s, t): return ('JUNK', t)


#### patterns
_NIKUD = u"\u05b0-\u05c4"
_TEAMIM = u"\u0591-\u05af"

undigraph = lambda x: x.replace(u"\u05f0", u"׳•׳•").replace(u"\u05f1", u"׳•׳™").replace("\u05f2", "׳™׳™").replace(
    "\ufb4f", "׳׳").replace(u"\u200d", "")

_heb_letter = r"([׳-׳×%s]|[׳“׳’׳–׳¦׳×׳˜]')" % _NIKUD

# a heb word including single quotes, dots and dashes  / this leaves last-dash out of the word
_heb_word_plus = r"[׳-׳×%s]([.'`\"\-/\\]?['`]?[׳-׳×%s0-9'`])*" % (_NIKUD, _NIKUD)

# english/latin words  (do not care about abbreviations vs. eos for english)
_eng_word = r"[a-zA-Z][a-zA-Z0-9'.]*"

# numerical expression (numbers and various separators)
_numeric = r"[+-]?([0-9][0-9.,/\-:]*)?[0-9]%?"

# url
_url = r"[a-z]+://\S+"

# punctuations
_opening_punc = r"[\[('`\"{]"
_closing_punc = r"[\])'`\"}]"
_eos_punct = r"[!?.]+"
_internal_punct = r"[,;:\-&]"

# junk
_junk = r"[^׳-׳×%sa-zA-Z0-9!?.,:;\-()\[\]{}]+" % _NIKUD  # %%&!?.,;:\-()\[\]{}\"'\/\\+]+" #% _NIKUD

is_all_heb = re.compile(r"^%s+$" % (_heb_letter), re.UNICODE).match
is_a_number = re.compile(r"^%s$" % _numeric, re.UNICODE).match
is_all_lat = re.compile(r"^[a-zA-Z]+$", re.UNICODE).match
is_sep = re.compile(r"^\|+$").match
is_punct = re.compile(r"^[.?!]+").match

#### scanner
scanner = re.Scanner([
    (r"\s+", None),
    (_url, url),
    (_heb_word_plus, heb),
    (_eng_word, eng),
    (_numeric, num),
    (_opening_punc, punct),
    (_closing_punc, punct),
    (_eos_punct, punct),
    (_internal_punct, punct),
    (_junk, junk),
])


##### tokenize
def tokenize(sent):
    tok = sent
    parts, reminder = scanner.scan(tok)
    tokens = []
    for he_word in parts:
        if he_word[0] == "HEB":
            tokens.append(he_word[1])
    assert (not reminder)
    return tokens


he_sw = pd.read_csv("./data/he_sp.csv", encoding="utf-8")
he_sw = list(he_sw["words"])


def stop_word_remover(text, is_split=True, return_split=True):
    if is_split and return_split:
        for word in text:
            if word in he_sw:
                text.remove(word)
        return text
    elif is_split and (not return_split):
        sent = ""
        for word in text:
            if word not in he_sw:
                sent = sent + " " + word
        return sent
    elif (not is_split) and return_split:
        output_tokens = []
        for word in text.split():
            if word not in he_sw:
                output_tokens.append(word)
        return output_tokens
    elif (not is_split) and (not return_split):
        sent = ""
        for word in text.split():
            if word not in he_sw:
                sent = sent + " " + word
        return sent




# s = "איש מוסד בגרמניה שהיה אחראי על איסוף מידע בנוגע למשרד שבו שהו אזרחים ערבים בטורקיה. הוא זיהה סטודנטים פלסטינים שהיו זקוקים לסיוע כספי והעביר את שמותיהם ל-AZ. הוא שוחח עם אותו  sfgrsdg n fsdgdf קצין מוסד באמצעות אפליקציות מבוססות רשת."
#
# f = stop_word_remover(tokenize(s),is_split=True, return_split=True)
#
#
# print(len(s.split()))
# print(len(tokenize(s)))
# print(f)
# print(len(f))