import requests
from bs4 import BeautifulSoup
import asyncio
import json
import re
import unicodedata

url = "https://animetosho.org/search"

def clean_title(t):
    t = unicodedata.normalize('NFKC', t)
    t = re.sub(
        r'[\u200B\u200C\u200D\u200E\u200F\uFEFF\u00AD\u2060\u180E\u034F]', 
        '', 
        t
    )
    t = ''.join(char for char in t if unicodedata.category(char) not in ('Cc', 'Cf'))
    return t

VIDEO_SOURCES = {'WEB-DL', 'WEBRIP', 'BD', 'BLURAY', 'DVD', 'HDTV'}
STREAMING_SERVICES = {'KKTV', 'CR', 'NF', 'AMZN', 'DSNP', 'BILI', 'HIDIVE', 'APPS', 'YTB'}
AUDIO_TERMS = {'AAC', 'AC3', 'DDP', 'FLAC', 'OPUS', 'DTS', 'TRUEHD', 'DUAL', 'MULTI', '2.0', '5.1', '7.1', 'ENGLISH DUB', 'DUB'}
CODEC_TERMS = {'HEVC', 'AVC', 'X264', 'X265', 'H.264', 'H.265', '10BIT', '8BIT'}
RES_TERMS = {'1080P', '720P', '480P', '2160P', '4K'}

def heuristic_labeler(tokens, i):
    token = tokens[i]
    upper_tok = token.upper()
    
    is_bracketed = token.startswith('[') and token.endswith(']')
    
    # Strip brackets for content inspection: "[1080p]" -> "1080P"
    clean_tok = token[1:-1] if is_bracketed else token
    upper_clean = clean_tok.upper()
    
    # --- PRIORITY 1: Content-based Checks (Even inside brackets) ---
    
    # 1. Check for Hash (Specific Hex Format inside brackets)
    if is_bracketed and re.match(r'^[0-9A-F]{8}$', upper_clean):
        return "HASH"
        
    # 2. Check Dictionaries (Matches "1080p", "Web-DL", "English Dub" etc.)
    if upper_clean in RES_TERMS: return "RES"
    if upper_clean in VIDEO_SOURCES: return "SOURCE"
    if upper_clean in STREAMING_SERVICES: return "SOURCE"
    if upper_clean in AUDIO_TERMS: return "AUDIO"
    if upper_clean in CODEC_TERMS: return "CODEC"

    # 3. Fuzzy match for Audio (e.g., "Dub" inside brackets)
    if "DUB" in upper_clean:
        return "AUDIO"

    # --- PRIORITY 2: Structural Checks ---
    
    # If it was bracketed but didn't match the above, it's likely a Group
    if is_bracketed:
        return "GROUP" 

    if token.startswith('(') and token.endswith(')'):
        return "META"

    # --- PRIORITY 3: Contextual Checks ---
    prev_tok = tokens[i-1].upper() if i > 0 else ""
    next_tok = tokens[i+1].upper() if i < len(tokens)-1 else ""
    
    # Handle Season/Episode Numbers
    if token.isdigit() or re.match(r'^\d{1,4}$', token):
        if prev_tok in ['S', 'SEASON']: return "SEASON"
        if prev_tok in ['E', 'EPISODE', '-']: return "EPISODE"
            
    if upper_tok in ['S', 'SEASON']: return "SEASON"
    if upper_tok in ['E', 'EPISODE']: return "EPISODE"
    
    # Handle Split Codecs (H . 264)
    if upper_tok == 'H' and (next_tok == '.' or next_tok in ['264', '265']):
        return "CODEC"
    if token == '.' and prev_tok == 'H' and next_tok in ['264', '265']:
        return "CODEC"
    if token in ['264', '265'] and (prev_tok == '.' or prev_tok == 'H' or prev_tok == 'X'):
        return "CODEC"

    # Punctuation vs Title
    if re.match(r'^[^\w\s]+$', token): 
        return "O"

    return "TITLE"

def tokenize(title):
    cleaned = clean_title(title)
    pattern = r"\[[^\]]+\]|" + \
              r"\([^\)]+\)|" + \
              r"\d{1,4}p|" + \
              r"\d+\.?\d*|" + \
              r"[A-Za-z]+(?:['-][A-Za-z]+)*|" + \
              r"[^\s]"
    
    tokens = re.findall(pattern, cleaned)
    return tokens

def createTrainingData(tokens):
    training_data = {
        'GROUP': [], 'SEASON': [], 'EPISODE': [], 
        'RES': [], 'SOURCE': [], 'AUDIO': [], 
        'CODEC': [], 'O': [], 'TITLE': [], 
        'HASH': [], 'META': []
    }
    
    for i in range(len(tokens)):
        label = heuristic_labeler(tokens, i)
        token = tokens[i]
        
        if label not in training_data:
            training_data[label] = []
            
        training_data[label].append(token)
    
    return training_data

async def main():
    try:
        response = requests.get(url, params={'q': 'overlord'})

        if response.status_code == 200:
            print("Request success")
            
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            
            results = []
            
            entries = soup.select('.home_list_entry.home_list_entry_alt, .home_list_entry, .home_list_entry_compl_1')
            
            for element in entries:
                title_element = element.select_one('.link a')
                title = title_element.text.strip() if title_element else None
                
                if title:
                    tokens = tokenize(title)
                    labeled_data = createTrainingData(tokens)
                    
                    results.append({
                        'title': title, 
                        'tokens': tokens, 
                        'training_data': labeled_data
                    })
            
            print(f"Found {len(results)} results")
            
            with open('dump1.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        else:
            print(f"Failed: {response.status_code}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())